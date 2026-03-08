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
from datasets import Dataset, Audio, Features, Value
from huggingface_hub import HfApi, login


def push_dataset(
    output_dir: str,
    repo_name: str,
    private: bool = False,
):
    """
    Push the generated dataset (metadata + audio files) to HuggingFace Hub.

    The script reads metadata_cloned.csv, resolves audio file paths,
    and creates a HuggingFace Dataset with Audio features.
    """
    # ── Load metadata ───────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "metadata_cloned.csv")
    if not os.path.exists(csv_path):
        # Try alternate name from MagpieTTS script
        csv_path = os.path.join(output_dir, "metadata_enhanced.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(output_dir, "metadata.csv")
    if not os.path.exists(csv_path):
        print(f"Error: No metadata CSV found in {output_dir}")
        sys.exit(1)

    print(f"Loading metadata from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")

    # ── Resolve audio paths to absolute paths ───────────────────────────────
    audio_columns = [c for c in df.columns if c.startswith("cloned_voice_")]

    for col in audio_columns:
        df[col] = df[col].apply(
            lambda x: os.path.join(output_dir, x) if pd.notna(x) and x != "" else None
        )

    # ── Build HuggingFace Dataset ───────────────────────────────────────────
    # Convert to dict format
    data_dict = df.to_dict(orient="list")

    # Create dataset
    ds = Dataset.from_dict(data_dict)

    # Cast audio columns to Audio type
    for col in audio_columns:
        try:
            ds = ds.cast_column(col, Audio())
            print(f"  Cast '{col}' as Audio feature")
        except Exception as e:
            print(f"  ⚠ Could not cast '{col}' as Audio: {e}")

    print(f"\nDataset ready: {ds}")
    print(f"  Features: {ds.features}")

    # ── Push to Hub ─────────────────────────────────────────────────────────
    print(f"\nPushing to HuggingFace Hub: {repo_name}")
    ds.push_to_hub(
        repo_name,
        private=private,
        commit_message="Add voice cloning dataset with multilingual audio",
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
