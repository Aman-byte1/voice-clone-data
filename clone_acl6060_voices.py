"""
Enhance the ACL 60/60 dataset (ymoslem/acl-6060) with cloned voice columns
in target languages using MagpieTTS.

The ACL 60/60 dataset already contains translations in multiple languages:
  text_en, text_ar, text_de, text_fa, text_fr, text_ja,
  text_nl, text_pt, text_ru, text_tr, text_zh

For each row, this script:
1. Reads the existing translated text from the dataset columns
2. Generates cloned speech in target languages using MagpieTTS
3. Saves audio files and creates an enhanced metadata CSV/JSON

Usage:
    python clone_acl6060_voices.py \\
        --output_dir ./output/acl6060_enhanced \\
        --target_languages fr,zh \\
        --split dev \\
        --max_rows 468

    # Quick test on first 5 rows:
    python clone_acl6060_voices.py \\
        --output_dir ./output/acl6060_enhanced \\
        --target_languages fr \\
        --max_rows 5
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

from tts_utils import (
    SUPPORTED_LANGUAGES,
    SPEAKER_NAMES,
    SPEAKER_MAP,
    ensure_dir,
    generate_speech,
    load_magpie_model,
    save_audio,
    DEFAULT_SAMPLE_RATE,
)


# ─── Dataset Configuration ─────────────────────────────────────────────────────

DATASET_NAME = "ymoslem/acl-6060"

# Mapping of language codes to the text column names in the ACL 60/60 dataset
ACL_TEXT_COLUMNS = {
    "en": "text_en",
    "ar": "text_ar",
    "de": "text_de",
    "fa": "text_fa",
    "fr": "text_fr",
    "ja": "text_ja",
    "nl": "text_nl",
    "pt": "text_pt",
    "ru": "text_ru",
    "tr": "text_tr",
    "zh": "text_zh",
}

# Languages supported by BOTH the ACL dataset AND MagpieTTS
# MagpieTTS: en, es, de, fr, vi, it, zh, hi, ja
# ACL 60/60: en, ar, de, fa, fr, ja, nl, pt, ru, tr, zh
# Overlap:   en, de, fr, ja, zh
OVERLAPPING_LANGUAGES = {"en", "de", "fr", "ja", "zh"}

DEFAULT_SPEAKER = "Sofia"


# ─── Main Logic ─────────────────────────────────────────────────────────────────


def load_acl_dataset(split: str = "dev", max_rows: int | None = None):
    """Load the ACL 60/60 dataset from HuggingFace."""
    print(f"Loading dataset: {DATASET_NAME} (split={split})")
    ds = load_dataset(DATASET_NAME, split=split)
    if max_rows and max_rows < len(ds):
        ds = ds.select(range(max_rows))
        print(f"  Selected first {max_rows} rows.")
    print(f"  Total rows: {len(ds)}")
    print(f"  Columns: {ds.column_names}")
    return ds


def enhance_dataset(
    output_dir: str,
    target_languages: list[str],
    split: str = "dev",
    max_rows: int | None = None,
    speaker: str = DEFAULT_SPEAKER,
    device: str = "cuda",
):
    """
    Enhance the ACL 60/60 dataset with cloned voice columns.

    For each row, uses the pre-existing translated text and generates
    speech via MagpieTTS in each target language.
    """
    audio_dir = ensure_dir(os.path.join(output_dir, "cloned_audio"))
    ds = load_acl_dataset(split=split, max_rows=max_rows)
    model = load_magpie_model(device=device)

    records = []
    total = len(ds) * len(target_languages)
    pbar = tqdm(total=total, desc="Generating cloned voices", unit="clip")

    for idx in range(len(ds)):
        row = ds[idx]

        # Build base record with source info
        row_record = {
            "index": row.get("index", idx),
            "source_text_en": row.get("text_en", ""),
        }

        # Copy all existing text translations for reference
        for lang_code, col_name in ACL_TEXT_COLUMNS.items():
            if col_name in row:
                row_record[f"text_{lang_code}"] = row[col_name]

        # Copy audio info if present
        audio_val = row.get("audio")
        if isinstance(audio_val, dict):
            row_record["source_audio_path"] = audio_val.get("path", "")

        # Generate cloned voice for each target language
        for target_lang in target_languages:
            lang_audio_dir = ensure_dir(os.path.join(audio_dir, target_lang))

            # Get the translated text from the dataset's existing column
            text_col = ACL_TEXT_COLUMNS.get(target_lang)
            translated_text = row.get(text_col, "") if text_col else ""

            if not translated_text:
                print(f"\n  ⚠ Row {idx}: no text for '{target_lang}', skipping.")
                row_record[f"cloned_voice_{target_lang}"] = ""
                row_record[f"cloned_length_{target_lang}"] = 0
                pbar.update(1)
                continue

            filename = f"cloned_{idx:05d}_{target_lang}.wav"
            filepath = os.path.join(lang_audio_dir, filename)

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

            pbar.update(1)

        records.append(row_record)

    pbar.close()

    # ── Save enhanced metadata ──────────────────────────────────────────────
    if records:
        df = pd.DataFrame(records)
        csv_path = os.path.join(output_dir, "metadata_enhanced.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved {len(records)} enhanced records to {csv_path}")

        json_path = os.path.join(output_dir, "metadata_enhanced.json")
        df.to_json(json_path, orient="records", force_ascii=False, indent=2)
        print(f"✓ Saved metadata JSON to {json_path}")

        # ── Summary ─────────────────────────────────────────────────────────
        print("\n── Enhanced Dataset Summary ──")
        print(f"  Source dataset: {DATASET_NAME} ({split})")
        print(f"  Total rows:      {len(records)}")
        print(f"  Target languages: {target_languages}")
        print(f"  Speaker used:    {speaker}")
        for lang in target_languages:
            col = f"cloned_voice_{lang}"
            filled = df[col].astype(bool).sum() if col in df.columns else 0
            print(f"  Cloned [{lang}]:    {filled}/{len(records)} successful")
        print(f"  Output dir:      {os.path.abspath(output_dir)}")
    else:
        print("\n⚠ No records were processed.")


# ─── CLI ────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enhance ACL 60/60 dataset with multilingual cloned voices using MagpieTTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Languages supported by BOTH ACL 60/60 AND MagpieTTS:
  de (German), fr (French), ja (Japanese), zh (Chinese)

Note: Arabic (ar) is in the ACL dataset but NOT supported by MagpieTTS.
      Use a different TTS model for Arabic voice cloning.
        """,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/acl6060_enhanced",
        help="Directory to save enhanced dataset files.",
    )
    parser.add_argument(
        "--target_languages",
        type=str,
        default="fr,zh",
        help="Comma-separated target language codes for voice cloning. "
             "Must be supported by both ACL 60/60 and MagpieTTS: de, fr, ja, zh.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Dataset split to use (dev or test).",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Max number of rows to process (None = all).",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default=DEFAULT_SPEAKER,
        choices=SPEAKER_NAMES,
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

    target_languages = [l.strip() for l in args.target_languages.split(",")]

    # Validate: must be supported by MagpieTTS
    for lang in target_languages:
        if lang not in SUPPORTED_LANGUAGES:
            print(
                f"Error: Language '{lang}' not supported by MagpieTTS. "
                f"Available: {list(SUPPORTED_LANGUAGES.keys())}"
            )
            sys.exit(1)

    # Warn if language not in ACL dataset
    for lang in target_languages:
        if lang not in ACL_TEXT_COLUMNS:
            print(
                f"Warning: Language '{lang}' has no text column in the ACL 60/60 dataset. "
                f"Available: {list(ACL_TEXT_COLUMNS.keys())}"
            )

    # Warn about non-overlapping languages
    non_overlap = [l for l in target_languages if l not in OVERLAPPING_LANGUAGES]
    if non_overlap:
        print(
            f"Warning: Languages {non_overlap} are supported by MagpieTTS but don't "
            f"have text columns in ACL 60/60. Only these overlap: {sorted(OVERLAPPING_LANGUAGES)}"
        )

    print("=" * 60)
    print("  ACL 60/60 Voice Cloning Enhancement")
    print("=" * 60)
    print(f"  Dataset:          {DATASET_NAME}")
    print(f"  Split:            {args.split}")
    print(f"  Target languages: {target_languages}")
    print(f"  Speaker:          {args.speaker}")
    print(f"  Max rows:         {args.max_rows or 'all'}")
    print(f"  Output dir:       {args.output_dir}")
    print(f"  Device:           {args.device}")
    print("=" * 60)

    enhance_dataset(
        output_dir=args.output_dir,
        target_languages=target_languages,
        split=args.split,
        max_rows=args.max_rows,
        speaker=args.speaker,
        device=args.device,
    )


if __name__ == "__main__":
    main()
