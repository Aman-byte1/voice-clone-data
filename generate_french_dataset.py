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
import random
import time
import subprocess
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# ─── Speaker Mapping ────────────────────────────────────────────────────────
# (original_split, start_idx, end_idx, speaker_id, speaker_name)
SPEAKER_MAP_DATA = [
    ("dev",   0,   106, 1, "Elena"),
    ("dev", 107, 188, 2, "Antoine"),
    ("dev", 189, 254, 3, "Michał"),
    ("dev", 255, 362, 4, "Jiawei"),
    ("dev", 363, 467, 5, "unknown_1"),
    ("eval",  0,  99, 6, "Allan"),
    ("eval", 100, 183, 7, "Antoine_MUV"),
    ("eval", 184, 239, 8, "unknown_2"),
    ("eval", 240, 329, 9, "Kamezawa"),
    ("eval", 330, 415, 10, "Asaf"),
]

def get_speaker_info(split, idx):
    """Find speaker_id and speaker_name for a given original split and index."""
    for s_split, start, end, s_id, name in SPEAKER_MAP_DATA:
        if split == s_split and start <= idx <= end:
            return f"speaker_{s_id:02d}", name
    return "speaker_unknown", "Unknown"


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

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    # The user is using ChatterboxMultilingualTTS now
    print(f"Loading Chatterbox Multilingual TTS model...")

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

    # ChatterboxMultilingualTTS.from_pretrained()
    _model = ChatterboxMultilingualTTS.from_pretrained(device=device)
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

    # Attach speaker metadata based on original split and index
    def map_speaker(example, idx, split_name):
        s_id, s_name = get_speaker_info(split_name, example.get("index", idx))
        example["speaker_id"] = s_id
        example["speaker_name"] = s_name
        example["original_split"] = split_name
        example["original_index"] = example.get("index", idx)
        # Composite key: unique across both original splits (dev/eval both start at 0)
        example["_resume_key"] = f"{split_name}:{example['original_index']}"
        return example

    ds_dev = ds_dev.map(lambda x, i: map_speaker(x, i, "dev"), with_indices=True)
    ds_eval = ds_eval.map(lambda x, i: map_speaker(x, i, "eval"), with_indices=True)

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


def process_row(idx, row, split_name, split_dir, audio_en_dir, audio_dir, model, pbar, device="cuda", language_id="fr", cfg_weight=0.0):
    """Process a single row for voice cloning."""
    # Use pre-mapped speaker metadata
    speaker_id = row.get("speaker_id", f"speaker_{idx:04d}")
    speaker_name = row.get("speaker_name", "Unknown")
    original_idx = row.get("original_index", idx)
    original_split = row.get("original_split", "unknown")
    resume_key = row.get("_resume_key", f"{original_split}:{original_idx}")
    
    row_record = {
        "_idx": idx,  # positional index for sorting
        "_resume_key": resume_key,
        "speaker": speaker_id,
        "speaker_name": speaker_name,
        "original_split": original_split,
        "original_index": str(original_idx),
        "text_en": row.get("text_en", ""),
        "text_fr": row.get("text_fr", ""),
        "audio_en": "",
        "cloned_audio_fr": "",
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
        row_record["audio_en"] = os.path.relpath(audio_en_filepath, split_dir)
        
        # Use a thread-safe way for temp files or just unique names
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        try:
            os.close(tmp_fd)
            sf.write(tmp_path, audio_array, sr)

            # Generate cloned French speech
            # NOTE: ChatterboxTTS is NOT thread-safe for concurrent inference on a single instance.
            # We use a lock to ensure only one thread generates at a time, while others handle I/O.
            with _model_lock:
                # NOTE: Exactly 0.0 can cause a tensor mismatch in some T3 versions.
                # We use a tiny epsilon to achieve the same effect safely.
                inference_cfg = cfg_weight if cfg_weight > 0 else 0.001
                
                with torch.no_grad():
                    wav = model.generate(
                        translated_text,
                        audio_prompt_path=tmp_path,
                        language_id=language_id,
                        cfg_weight=inference_cfg
                    )
                
                # Cleanup GPU memory immediately after generation
                if device == "cuda":
                    torch.cuda.empty_cache()

            # Save output
            ta.save(filepath, wav, model.sr)
            row_record["cloned_audio_fr"] = os.path.relpath(filepath, split_dir)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception:
        row_record["error"] = traceback.format_exc()

    pbar.update(1)
    return row_record

def trigger_upload(output_dir, repo_name):
    """Trigger the push_to_hub.py script to update HuggingFace."""
    if not repo_name:
        return
    print(f"\n[Incremental Upload] Pushing to {repo_name}...")
    try:
        subprocess.run(
            [sys.executable, "push_to_hub.py", "--output_dir", output_dir, "--repo_name", repo_name],
            check=True
        )
        print(f"[Incremental Upload] ✓ Sync complete.")
    except Exception as e:
        print(f"[Incremental Upload] ⚠ Push failed: {e}")


def generate_split(ds, split_name, output_dir, model, device, num_workers=8, language_id="fr", cfg_weight=0.0, repo_name=None, checkpoint_pct=None):
    """Generate French cloned audio for a single split using parallel threads. Supports Resuming."""
    split_dir = ensure_dir(os.path.join(output_dir, split_name))
    audio_en_dir = ensure_dir(os.path.join(split_dir, "original_audio_en"))
    audio_dir = ensure_dir(os.path.join(split_dir, "cloned_audio_fr"))

    csv_path = os.path.join(split_dir, "metadata_cloned.csv")
    records = []
    
    # --- Resume Logic ---
    if os.path.exists(csv_path):
        try:
            old_df = pd.read_csv(csv_path).fillna("")
            # A row is complete if it has a cloned_audio_fr path and the file exists
            def is_complete(row):
                if not row.get("cloned_audio_fr"): return False
                return os.path.exists(os.path.join(split_dir, row["cloned_audio_fr"]))

            completed_df = old_df[old_df.apply(is_complete, axis=1)]
            records = completed_df.to_dict(orient="records")
            # Use composite key (original_split:original_index) to avoid collisions
            # between dev and eval splits which both start at index 0
            if "_resume_key" in completed_df.columns:
                completed_keys = set(completed_df["_resume_key"].astype(str))
            else:
                # Backward compat: build key from split+index columns
                completed_keys = set(
                    completed_df.apply(
                        lambda r: f"{r['original_split']}:{r['original_index']}", axis=1
                    )
                )
            print(f"  ➜ Resuming [{split_name}]: Skipping {len(completed_keys)} already completed clips.")
        except Exception as e:
            print(f"  ⚠ Resume failed (starting split fresh): {e}")
            records = []
            completed_keys = set()
    else:
        completed_keys = set()

    pbar = tqdm(total=len(ds), desc=f"Cloning French [{split_name}]", unit="clip")
    pbar.update(len(completed_keys))

    total = len(ds)
    checkpoint_step = max(1, int(total * (checkpoint_pct / 100.0))) if checkpoint_pct else None
    
    # Identify which indices to process
    indices_to_process = []
    for idx in range(total):
        row_data = ds[idx]
        # Use composite key to uniquely identify rows across both original splits
        resume_key = row_data.get("_resume_key", f"{row_data.get('original_split', 'unknown')}:{row_data.get('original_index', idx)}")
        if resume_key not in completed_keys:
            indices_to_process.append(idx)
    
    if indices_to_process:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    process_row, idx, ds[idx], split_name, split_dir,
                    audio_en_dir, audio_dir, model, pbar,
                    device=device,
                    language_id=language_id,
                    cfg_weight=cfg_weight
                ): idx 
                for idx in indices_to_process
            }
            
            count = len(completed_keys)
            for future in as_completed(futures):
                res = future.result()
                if res:
                    records.append(res)
                count += 1
                
                # Periodic checkpoint upload
                if checkpoint_step and count % checkpoint_step == 0 and count < total:
                    # Save partial CSV so the uploader sees current progress
                    temp_df = pd.DataFrame([r for r in records if r is not None])
                    if "_idx" in temp_df.columns:
                        temp_df = temp_df.drop(columns=["_idx", "_resume_key"], errors="ignore")
                    temp_df.to_csv(csv_path, index=False)
                    trigger_upload(output_dir, repo_name)
    else:
        print(f"  ✓ Split [{split_name}] already 100% complete.")

    pbar.close()

    # Save metadata — sort by positional index for deterministic output
    if records:
        df = pd.DataFrame(records)
        if "_idx" in df.columns:
            df = df.sort_values("_idx").reset_index(drop=True)
            df = df.drop(columns=["_idx", "_resume_key"], errors="ignore")
        csv_path = os.path.join(split_dir, "metadata_cloned.csv")
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved {len(records)} records to {csv_path}")

        json_path = os.path.join(split_dir, "metadata_cloned.json")
        df.to_json(json_path, orient="records", force_ascii=False, indent=2)
        print(f"  ✓ Saved JSON to {json_path}")

        filled = df["cloned_audio_fr"].astype(bool).sum()
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
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (e.g. 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--language_id",
        type=str,
        default="fr",
        help="Target language ID (e.g. 'fr' for French, 'en' for English).",
    )
    parser.add_argument(
        "--cfg_weight",
        type=float,
        default=0.1,
        help="CFG weight (0.1 recommended for strong accent mitigation).",
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        default=None,
        help="Optional: HuggingFace repo name for incremental uploads.",
    )
    parser.add_argument(
        "--checkpoint_pct",
        type=int,
        default=None,
        help="Optional: Upload to HF every X percent of completion.",
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
    
    # Always upload after test set if repo_name is provided
    if args.repo_name:
        trigger_upload(args.output_dir, args.repo_name)

    print(f"\n── Generating TRAIN split ──")
    generate_split(ds_train, "train", args.output_dir, model, args.device, 
                   num_workers=args.num_workers, language_id=args.language_id, 
                   cfg_weight=args.cfg_weight, repo_name=args.repo_name, 
                   checkpoint_pct=args.checkpoint_pct)

    # Final upload to ensure 100% completion is pushed
    if args.repo_name:
        trigger_upload(args.output_dir, args.repo_name)

    print("\n" + "=" * 60)
    print("  ✓ All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
