"""
Script to generate French cloned voice dataset for a combined, specifically seeded subset.

This script combines the train and dev splits of ymoslem/acl-6060 dataset,
shuffles them with random seed 0, selects 100 rows, and runs MagpieTTS for French.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

from tts_utils import (
    ensure_dir,
    generate_speech,
    load_magpie_model,
    save_audio,
)

DATASET_NAME = "ymoslem/acl-6060"
DEFAULT_SPEAKER = "Sofia"
NUM_ROWS = 100
RANDOM_SEED = 0

def load_and_prepare_dataset():
    print(f"Loading {DATASET_NAME} dataset...")
    # The dataset has 'train' and 'dev' splits (per huggingface page, dev has 468 rows, train has others, totaling 884)
    ds_dev = load_dataset(DATASET_NAME, split="dev")
    ds_train = load_dataset(DATASET_NAME, split="eval")
    
    print(f"Loaded 'dev' split: {len(ds_dev)} rows.")
    print(f"Loaded 'train' split: {len(ds_train)} rows.")
    
    # Merge splits
    ds_combined = concatenate_datasets([ds_train, ds_dev])
    print(f"Combined total: {len(ds_combined)} rows.")
    
    # Shuffle and select 100 rows with seed 0
    ds_selected = ds_combined.shuffle(seed=RANDOM_SEED).select(range(NUM_ROWS))
    print(f"Selected {len(ds_selected)} rows after shuffling with seed {RANDOM_SEED}.")
    return ds_selected

def enhance_dataset(
    output_dir: str,
    speaker: str = DEFAULT_SPEAKER,
    device: str = "cuda",
):
    audio_dir = ensure_dir(os.path.join(output_dir, "cloned_audio_fr_test"))
    ds = load_and_prepare_dataset()
    model = load_magpie_model(device=device)

    records = []
    pbar = tqdm(total=len(ds), desc="Generating cloned French voices", unit="clip")
    
    target_lang = "fr"

    for idx in range(len(ds)):
        row = ds[idx]

        # Use index from original dataset + new sequence id
        row_record = {
            "selected_id": idx,
            "original_index": row.get("index", ""),
            "source_text_en": row.get("text_en", ""),
            "text_fr": row.get("text_fr", ""),
        }

        # Original audio if present
        audio_val = row.get("audio")
        if isinstance(audio_val, dict):
            row_record["source_audio_path"] = audio_val.get("path", "")

        translated_text = row.get("text_fr", "")

        if not translated_text:
            print(f"\n  ⚠ Row {idx}: no text for 'fr', skipping.")
            row_record[f"cloned_voice_{target_lang}"] = ""
            row_record[f"cloned_length_{target_lang}"] = 0
            pbar.update(1)
            records.append(row_record)
            continue

        filename = f"cloned_{idx:05d}_{target_lang}.wav"
        filepath = os.path.join(audio_dir, filename)

        try:
            audio, length = generate_speech(
                model, translated_text, target_lang, speaker=speaker
            )
            save_audio(audio, filepath)

            row_record[f"cloned_voice_{target_lang}"] = os.path.relpath(
                filepath, output_dir
            )
            row_record[f"cloned_length_{target_lang}"] = length

        except Exception as e:
            print(f"\n  ✗ Failed row {idx} -> {target_lang}: {e}")
            row_record[f"cloned_voice_{target_lang}"] = ""
            row_record[f"cloned_length_{target_lang}"] = 0

        records.append(row_record)
        pbar.update(1)

    pbar.close()

    # Save enhanced metadata
    if records:
        df = pd.DataFrame(records)
        csv_path = os.path.join(output_dir, "metadata_fr_test.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved {len(records)} test records to {csv_path}")

        json_path = os.path.join(output_dir, "metadata_fr_test.json")
        df.to_json(json_path, orient="records", force_ascii=False, indent=2)
        print(f"✓ Saved metadata JSON to {json_path}")
    else:
        print("\n⚠ No records were processed.")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a combined, shuffled 100-sample test set for French voice cloning."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/acl6060_test",
        help="Directory to save the test dataset files.",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default=DEFAULT_SPEAKER,
        help="MagpieTTS speaker to use for cloned voices.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 60)
    print("  Generating French Test Dataset")
    print("=" * 60)
    print(f"  Dataset:          {DATASET_NAME}")
    print(f"  Target language:  fr")
    print(f"  Num rows:         {NUM_ROWS}")
    print(f"  Random seed:      {RANDOM_SEED}")
    print(f"  Speaker:          {args.speaker}")
    print(f"  Output dir:       {args.output_dir}")
    print(f"  Device:           {args.device}")
    print("=" * 60)

    enhance_dataset(
        output_dir=args.output_dir,
        speaker=args.speaker,
        device=args.device,
    )

if __name__ == "__main__":
    main()
