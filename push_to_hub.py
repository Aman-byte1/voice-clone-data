"""
Push the generated voice cloning dataset to HuggingFace Hub.

Usage:
    python push_to_hub.py \\
        --output_dir ./output/acl6060_scicom \\
        --repo_name amanuelbyte/acl6060-voice-cloning-multilingual \\
        --private

    # Or just specify the metadata CSV:
    python push_to_hub.py \\
        --metadata_csv ./output/acl6060_scicom/metadata_cloned.csv \\
        --audio_dir ./output/acl6060_scicom/cloned_audio \\
        --repo_name amanuelbyte/acl6060-voice-cloning-multilingual

Requires:
    export HUGGING_FACE_HUB_TOKEN=hf_xxxxx
"""

import argparse
import os
import sys

import pandas as pd
from datasets import Dataset, DatasetDict, Audio
from huggingface_hub import HfApi, login


def push_dataset(
    output_dir: str,
    repo_name: str,
    private: bool = False,
):
    """
    Push the generated dataset splits (metadata + audio files) to HuggingFace Hub.
    Automatically finds all split subdirectories in output_dir.
    """
    dataset_dict = DatasetDict()

    if not os.path.exists(output_dir):
        print(f"Error: Directory {output_dir} does not exist.")
        sys.exit(1)

    # ── Discover and process each split ─────────────────────────────────────
    found_splits = []
    
    # Also check if the output_dir itself has a metadata.csv (single split legacy mode)
    possible_csvs = ["metadata_cloned.csv", "metadata_enhanced.csv", "metadata.csv"]
    for csv in possible_csvs:
        if os.path.exists(os.path.join(output_dir, csv)):
            found_splits.append(("train", output_dir, os.path.join(output_dir, csv)))
            break
            
    # If no root metadata, look in subdirectories for actual splits
    if not found_splits:
        for item in os.listdir(output_dir):
            split_dir = os.path.join(output_dir, item)
            if os.path.isdir(split_dir):
                for csv in possible_csvs:
                    csv_path = os.path.join(split_dir, csv)
                    if os.path.exists(csv_path):
                        found_splits.append((item, split_dir, csv_path))
                        break

    if not found_splits:
        print(f"Error: No valid data splits found in {output_dir}")
        sys.exit(1)

    for split_name, split_dir, csv_path in found_splits:
        print(f"\nProcessing split: {split_name}")
        print(f"  Loading metadata from {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")

        # Resolve audio paths to absolute paths
        audio_columns = [c for c in df.columns if c.startswith("cloned_voice_")]
        for col in audio_columns:
            df[col] = df[col].apply(
                lambda x: os.path.join(split_dir, x) if pd.notna(x) and x != "" else None
            )

        # Build Dataset
        ds = Dataset.from_dict(df.to_dict(orient="list"))
        for col in audio_columns:
            try:
                ds = ds.cast_column(col, Audio())
            except Exception as e:
                print(f"  ⚠ Could not cast '{col}' as Audio: {e}")

        dataset_dict[split_name] = ds
        print(f"  ✓ Added split '{split_name}' with {len(ds)} rows")

    print(f"\nDatasetDict ready: {dataset_dict}")

    # ── Push to Hub ─────────────────────────────────────────────────────────
    print(f"\nPushing to HuggingFace Hub: {repo_name} (splits: {list(dataset_dict.keys())})")
    dataset_dict.push_to_hub(
        repo_name,
        private=private,
        commit_message=f"Add voice cloning dataset ({', '.join(dataset_dict.keys())} splits)",
    )
    print(f"✓ Dataset pushed successfully to https://huggingface.co/datasets/{repo_name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Push generated voice cloning dataset to HuggingFace Hub.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory containing the generated dataset (metadata CSV + audio files).",
    )
    parser.add_argument(
        "--repo_name", type=str, required=True,
        help="HuggingFace dataset repo name (e.g., 'amanuelbyte/acl6060-voice-cloning-multilingual').",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Make the dataset repo private.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Check HF token
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("✓ Logged in to HuggingFace Hub")
    else:
        print("⚠ No HF token found in environment. Trying cached credentials...")

    push_dataset(
        output_dir=args.output_dir,
        repo_name=args.repo_name,
        private=args.private,
    )


if __name__ == "__main__":
    main()
