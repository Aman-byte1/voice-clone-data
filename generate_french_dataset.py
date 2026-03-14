"""
Generate French cloned voice dataset with train/test splits.

Combines both splits of ymoslem/acl-6060 (884 total segments),
shuffles with seed 0, selects 100 for test and 784 for train,
then generates French cloned audio using MagpieTTS.

Usage (on server):
    python generate_french_dataset.py \
        --output_dir ./output/acl6060_fr \
        --device cuda
"""

import argparse
import os
import sys

import pandas as pd
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

from tts_utils import (
    ensure_dir,
    generate_speech,
    load_magpie_model,
    save_audio,
)

DATASET_NAME = "ymoslem/acl-6060"
TTS_MODEL_NAME = "nvidia/magpie_tts_multilingual_357m"
DEFAULT_SPEAKER = "Sofia"
NUM_TEST = 100
RANDOM_SEED = 0


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


def generate_split(ds, split_name, output_dir, model, speaker):
    """Generate French cloned audio for a single split."""
    split_dir = ensure_dir(os.path.join(output_dir, split_name))
    audio_dir = ensure_dir(os.path.join(split_dir, "cloned_audio_fr"))

    target_lang = "fr"
    records = []
    pbar = tqdm(
        total=len(ds),
        desc=f"Generating French [{split_name}]",
        unit="clip",
    )

    for idx in range(len(ds)):
        row = ds[idx]

        row_record = {
            "selected_id": idx,
            "original_index": row.get("index", ""),
            "source_text_en": row.get("text_en", ""),
            "text_fr": row.get("text_fr", ""),
            "tts_model": TTS_MODEL_NAME,
            "speaker": speaker,
        }

        # Keep source audio path if present
        audio_val = row.get("audio")
        if isinstance(audio_val, dict):
            row_record["source_audio_path"] = audio_val.get("path", "")

        translated_text = row.get("text_fr", "")

        if not translated_text:
            print(f"\n  ⚠ Row {idx}: no French text, skipping.")
            row_record["cloned_voice_fr"] = ""
            row_record["cloned_length_fr"] = 0
            records.append(row_record)
            pbar.update(1)
            continue

        filename = f"cloned_{idx:05d}_fr.wav"
        filepath = os.path.join(audio_dir, filename)

        try:
            audio, length = generate_speech(
                model, translated_text, target_lang, speaker=speaker
            )
            save_audio(audio, filepath)
            row_record["cloned_voice_fr"] = os.path.relpath(filepath, split_dir)
            row_record["cloned_length_fr"] = length
        except Exception as e:
            print(f"\n  ✗ Failed row {idx} -> fr: {e}")
            row_record["cloned_voice_fr"] = ""
            row_record["cloned_length_fr"] = 0

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

        filled = df["cloned_voice_fr"].astype(bool).sum()
        print(f"  ✓ Successful clones: {filled}/{len(records)}")
    else:
        print(f"  ⚠ No records for {split_name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate French cloned voice dataset (train + test splits)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/acl6060_fr",
        help="Root output directory.",
    )
    parser.add_argument(
        "--speaker", type=str, default=DEFAULT_SPEAKER,
        help="MagpieTTS speaker name.",
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
    print(f"  Source dataset:  {DATASET_NAME}")
    print(f"  TTS model:       {TTS_MODEL_NAME}")
    print(f"  Target language: fr (French)")
    print(f"  Random seed:     {RANDOM_SEED}")
    print(f"  Test samples:    {NUM_TEST}")
    print(f"  Speaker:         {args.speaker}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Device:          {args.device}")
    print("=" * 60)

    ds_train, ds_test = load_and_split_dataset()
    model = load_magpie_model(device=args.device)

    print("\n── Generating TEST split ──")
    generate_split(ds_test, "test", args.output_dir, model, args.speaker)

    print("\n── Generating TRAIN split ──")
    generate_split(ds_train, "train", args.output_dir, model, args.speaker)

    print("\n" + "=" * 60)
    print("  ✓ All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
