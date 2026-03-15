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

import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio as ta
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

DATASET_NAME = "ymoslem/acl-6060"
TTS_MODEL_NAME = "resemble-ai/chatterbox-multilingual"
NUM_TEST = 100
RANDOM_SEED = 0


# ─── Model Loading ─────────────────────────────────────────────────────────────

_model = None


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
        import torch
        original_load = torch.load
        def patched_load(*args, **kwargs):
            if 'map_location' not in kwargs:
                kwargs['map_location'] = 'cpu'
            return original_load(*args, **kwargs)
        torch.load = patched_load
        print("✓ Applied torch.load monkeypatch for CPU execution.")

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    import chatterbox.mtl_tts as mtl_tts

    # Aggressively mock Resemble Watermarker AFTER importing chatterbox
    # The real resemble_perth sets its class to None if CUDA isn't available
    class MockWatermarker:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return self
        def encode(self, wav, *args, **kwargs): return wav
        def __getattr__(self, name): return lambda *a, **k: None

    class MockPerth:
        PerthImplicitWatermarker = MockWatermarker

    mtl_tts.perth = MockPerth()
    print("✓ Forcefully applied Resemble Watermarker mock on mtl_tts.perth.")

    print(f"Loading Chatterbox Multilingual TTS model...")
    _model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    print("Model loaded successfully.")
    return _model


# ─── Dataset Loading ───────────────────────────────────────────────────────────


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_and_split_dataset():
    """Load both splits, merge, shuffle with seed 0, split into train/test."""
    print(f"Loading {DATASET_NAME} dataset...")
    ds_dev = load_dataset(DATASET_NAME, split="dev")
    ds_eval = load_dataset(DATASET_NAME, split="eval")

    print(f"  'dev' split:  {len(ds_dev)} rows")
    print(f"  'eval' split: {len(ds_eval)} rows")

    ds_combined = concatenate_datasets([ds_dev, ds_eval])
    print(f"  Combined:     {len(ds_combined)} rows")

    ds_shuffled = ds_combined.shuffle(seed=RANDOM_SEED)
    ds_test = ds_shuffled.select(range(NUM_TEST))
    ds_train = ds_shuffled.select(range(NUM_TEST, len(ds_shuffled)))

    print(f"  Test split:   {len(ds_test)} rows")
    print(f"  Train split:  {len(ds_train)} rows")
    return ds_train, ds_test


# ─── Generation ────────────────────────────────────────────────────────────────


def generate_split(ds, split_name, output_dir, model, device):
    """Generate French cloned audio for a single split using Chatterbox voice cloning."""
    split_dir = ensure_dir(os.path.join(output_dir, split_name))
    audio_en_dir = ensure_dir(os.path.join(split_dir, "original_audio_en"))
    audio_dir = ensure_dir(os.path.join(split_dir, "cloned_audio_fr"))

    target_lang = "fr"
    records = []
    pbar = tqdm(total=len(ds), desc=f"Cloning French [{split_name}]", unit="clip")

    for idx in range(len(ds)):
        row = ds[idx]

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
            print(f"\n  ⚠ Row {idx}: error accessing audio data: {e}. Skipping.")

        if audio_array is None or sr is None or len(audio_array) == 0:
            print(f"\n  ⚠ Row {idx}: no valid source audio data found (type: {type(audio_val)}). Skipping.")
            records.append(row_record)
            pbar.update(1)
            continue

        translated_text = row.get("text_fr", "")
        if not translated_text:
            print(f"\n  ⚠ Row {idx}: no French text, skipping.")
            records.append(row_record)
            pbar.update(1)
            continue

        # Save source audio to a temp file for Chatterbox voice cloning prompt
        # Also save it permanently to our audio_en directory

        audio_en_filename = f"original_{idx:05d}_en.wav"
        audio_en_filepath = os.path.join(audio_en_dir, audio_en_filename)
        
        filename = f"cloned_{idx:05d}_fr.wav"
        filepath = os.path.join(audio_dir, filename)

        try:
            # Write reference audio permanently
            sf.write(audio_en_filepath, audio_array, sr)
            row_record["auido_en"] = os.path.relpath(audio_en_filepath, split_dir)
            
            # Write reference audio to a temp file for model specific format needs
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, audio_array, sr)

            # Generate cloned French speech
            wav = model.generate(
                translated_text,
                audio_prompt_path=tmp_path,
                language_id=target_lang,
            )

            # Save output
            ta.save(filepath, wav, model.sr)
            row_record["cloned_auido_fr"] = os.path.relpath(filepath, split_dir)

        except Exception as e:
            print(f"\n  ✗ Failed row {idx} -> fr: {e}")
        finally:
            # Clean up temp file
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        records.append(row_record)
        pbar.update(1)

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
        "--device", type=str, default="cuda", choices=["cuda", "cpu"],
        help="Device for inference.",
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
    print(f"  Test samples:    {NUM_TEST}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Device:          {args.device}")
    print("=" * 60)

    ds_train, ds_test = load_and_split_dataset()
    model = load_model(device=args.device)

    print("\n── Generating TEST split ──")
    generate_split(ds_test, "test", args.output_dir, model, args.device)

    print("\n── Generating TRAIN split ──")
    generate_split(ds_train, "train", args.output_dir, model, args.device)

    print("\n" + "=" * 60)
    print("  ✓ All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
