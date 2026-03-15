"""
Generate French cloned voice dataset with train/test splits.

Combines both splits of ymoslem/acl-6060 (884 total segments),
shuffles with seed 0, selects 100 for test and 784 for train,
then generates French cloned audio using Chatterbox Multilingual TTS.

Usage (on server):
    python generate_french_dataset.py \
        --output_dir ./output/acl6060_fr \
        --device cuda
"""

import argparse
import os
import sys
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio as ta
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import tqdm as tqdm_base
from functools import partial

# Force tqdm to always show a bar with a fixed width, even in non-interactive terminals
tqdm_base.tqdm = partial(tqdm_base.tqdm, dynamic_ncols=False, ncols=100, ascii=True)

DATASET_NAME = "ymoslem/acl-6060"
TTS_MODEL_NAME = "resemble-ai/chatterbox-multilingual"
NUM_TEST = 100
RANDOM_SEED = 0


# ─── Model Loading ─────────────────────────────────────────────────────────────

_model = None
_model_lock = threading.Lock()


def load_model(device: str = "cuda"):
    """Load the Chatterbox Multilingual TTS model."""
    global _model

    if _model is not None:
        return _model

    # Auto-fallback if cuda requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    # Monkeypatch torch.load to handle CPU-only environments for Chatterbox checkpoints
    if device == "cpu":
        original_load = torch.load
        def patched_load(*args, **kwargs):
            if 'map_location' not in kwargs:
                kwargs['map_location'] = 'cpu'
            return original_load(*args, **kwargs)
        torch.load = patched_load
        print("✓ Applied torch.load monkeypatch for CPU execution.")

    from chatterbox.tts import ChatterboxTTS

    # The user is using ChatterboxTTS now
    print(f"Loading Chatterbox TTS model...")

    # Python 3.13 / RunPod: resemble-perth often fails to load its watermarker,
    # leading to TypeError: 'NoneType' object is not callable.
    try:
        import perth
        if not hasattr(perth, "PerthImplicitWatermarker") or perth.PerthImplicitWatermarker is None:
            print("⚠ resemble-perth is broken. Applying dummy watermarker monkeypatch.")
            class DummyWatermarker:
                def __init__(self, *args, **kwargs): pass
                def __call__(self, audio, *args, **kwargs): return audio
            perth.PerthImplicitWatermarker = DummyWatermarker
    except ImportError:
        pass

    # ChatterboxTTS.from_pretrained() often doesn't like attn_implementation in stable releases
    _model = ChatterboxTTS.from_pretrained(device=device)
    print("Model loaded successfully.")
    return _model


# ─── Dataset Loading ───────────────────────────────────────────────────────────


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_and_split_dataset(num_train=None, num_test=NUM_TEST):
    """Load both splits, merge, shuffle with seed 0, split into train/test."""
    print(f"Loading {DATASET_NAME} dataset...")
    ds_dev = load_dataset(DATASET_NAME, split="dev")
    ds_eval = load_dataset(DATASET_NAME, split="eval")

    print(f"  'dev' split:  {len(ds_dev)} rows")
    print(f"  'eval' split: {len(ds_eval)} rows")

    ds_combined = concatenate_datasets([ds_dev, ds_eval])
    print(f"  Combined:     {len(ds_combined)} rows")

    ds_shuffled = ds_combined.shuffle(seed=RANDOM_SEED)
    
    # Selection logic
    ds_test = ds_shuffled.select(range(num_test))
    
    # If num_train is specified, select exactly that many after the test set
    # Otherwise select everything else
    if num_train is not None:
        ds_train = ds_shuffled.select(range(num_test, num_test + num_train))
    else:
        ds_train = ds_shuffled.select(range(num_test, len(ds_shuffled)))

    print(f"  Test split:   {len(ds_test)} rows")
    print(f"  Train split:  {len(ds_train)} rows")
    return ds_train, ds_test


# ─── Generation ────────────────────────────────────────────────────────────────


def process_row(idx, row, split_name, split_dir, audio_en_dir, audio_dir, model, pbar, language_id="fr", cfg_weight=0.0):
    """Process a single row for voice cloning."""
    row_record = {
        "speaker": str(row.get("index", f"speaker_{idx}")),
        "text_en": row.get("text_en", ""),
        "fr_text": row.get("text_fr", ""),
        "auido_en": "",
        "cloned_auido_fr": "",
    }

    # Get source audio for voice cloning
    audio_val = row.get("audio")
    
    # Robust audio loading: handle dict or AudioDecoder objects
    audio_array = None
    sr = None

    try:
        if isinstance(audio_val, dict):
            audio_array = np.array(audio_val.get("array"), dtype=np.float32)
            sr = audio_val.get("sampling_rate")
        elif audio_val is not None:
            # Try accessing as dict-like or object-like (for AudioDecoder)
            try:
                audio_array = np.array(audio_val["array"], dtype=np.float32)
                sr = audio_val["sampling_rate"]
            except (KeyError, TypeError):
                audio_array = np.array(getattr(audio_val, "array", None), dtype=np.float32)
                sr = getattr(audio_val, "sampling_rate", None)
    except Exception as e:
        row_record["error"] = f"Audio access error: {e}"

    if audio_array is None or sr is None or len(audio_array) == 0:
        pbar.update(1)
        return row_record

    translated_text = row.get("text_fr", "")
    if not translated_text:
        row_record["error"] = "No French text"
        pbar.update(1)
        return row_record

    audio_en_filename = f"original_{idx:05d}_en.wav"
    audio_en_filepath = os.path.join(audio_en_dir, audio_en_filename)
    
    filename = f"cloned_{idx:05d}_fr.wav"
    filepath = os.path.join(audio_dir, filename)

    try:
        # Write reference audio permanently
        sf.write(audio_en_filepath, audio_array, sr)
        row_record["auido_en"] = os.path.relpath(audio_en_filepath, split_dir)
        
        # Use a thread-safe way for temp files or just unique names
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        try:
            os.close(tmp_fd)
            sf.write(tmp_path, audio_array, sr)

            # Generate cloned French speech
            # NOTE: ChatterboxTTS is NOT thread-safe for concurrent inference on a single instance.
            # We use a lock to ensure only one thread generates at a time, while others handle I/O.
            with _model_lock:
                wav = model.generate(
                    translated_text,
                    audio_prompt_path=tmp_path,
                    language_id=language_id,
                    cfg_weight=cfg_weight
                )

            # Save output
            ta.save(filepath, wav, model.sr)
            row_record["cloned_auido_fr"] = os.path.relpath(filepath, split_dir)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception:
        row_record["error"] = traceback.format_exc()

    pbar.update(1)
    return row_record


def generate_split(ds, split_name, output_dir, model, device, num_workers=8, language_id="fr", cfg_weight=0.0):
    """Generate French cloned audio for a single split using parallel threads."""
    split_dir = ensure_dir(os.path.join(output_dir, split_name))
    audio_en_dir = ensure_dir(os.path.join(split_dir, "original_audio_en"))
    audio_dir = ensure_dir(os.path.join(split_dir, "cloned_audio_fr"))

    pbar = tqdm(total=len(ds), desc=f"Cloning French [{split_name}]", unit="clip")

    # Use ThreadPoolExecutor for parallel processing
    # Multi-threading is often efficient for Torch inference as it releases the GIL
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_row, idx, ds[idx], split_name, split_dir,
                audio_en_dir, audio_dir, model, pbar,
                language_id=language_id,
                cfg_weight=cfg_weight
            )
            for idx in range(len(ds))
        ]
        records = [f.result() for f in futures]

    pbar.close()

    # Save metadata
    if records:
        df = pd.DataFrame(records)
        csv_path = os.path.join(split_dir, "metadata_cloned.csv")
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved {len(records)} records to {csv_path}")

        json_path = os.path.join(split_dir, "metadata_cloned.json")
        df.to_json(json_path, orient="records", force_ascii=False, indent=2)
        print(f"  ✓ Saved JSON to {json_path}")

        filled = df["cloned_auido_fr"].astype(bool).sum()
        print(f"  ✓ Successful clones: {filled}/{len(records)}")
    else:
        print(f"  ⚠ No records for {split_name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate French cloned voice dataset (train + test) using Chatterbox TTS."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/acl6060_fr",
        help="Root output directory.",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=NUM_TEST,
        help=f"Number of samples for the test split (default: {NUM_TEST}).",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=None,
        help="Number of samples for the train split (default: all remaining).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel worker threads (default: 4).",
    )
    parser.add_argument(
        "--language_id",
        type=str,
        default="fr",
        help="Language ID for cloning (e.g. 'fr', 'en'). Use 'fr' for French.",
    )
    parser.add_argument(
        "--cfg_weight",
        type=float,
        default=0.0,
        help="CFG weight (0.0 recommended for cross-lingual to mitigate accent).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  French Voice Cloning — Train + Test Splits")
    print("=" * 60)
    print(f"  Source dataset:   {DATASET_NAME}")
    print(f"  TTS model:       {TTS_MODEL_NAME}")
    print(f"  Target language: fr (French)")
    print(f"  Random seed:     {RANDOM_SEED}")
    print(f"  Test samples:    {args.num_test}")
    print(f"  Train samples:   {args.num_train or 'all remaining'}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Device:          {args.device}")
    print("=" * 60)

    ds_train, ds_test = load_and_split_dataset(num_train=args.num_train, num_test=args.num_test)
    model = load_model(device=args.device)

    print(f"\n── Generating TEST split ──")
    generate_split(ds_test, "test", args.output_dir, model, args.device, 
                   num_workers=args.num_workers, language_id=args.language_id, 
                   cfg_weight=args.cfg_weight)

    print(f"\n── Generating TRAIN split ──")
    generate_split(ds_train, "train", args.output_dir, model, args.device, 
                   num_workers=args.num_workers, language_id=args.language_id, 
                   cfg_weight=args.cfg_weight)

    print("\n" + "=" * 60)
    print("  ✓ All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
