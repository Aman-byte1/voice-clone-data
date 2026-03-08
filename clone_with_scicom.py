"""
Clone voices from ACL datasets using Scicom Multilingual-TTS-1.7B-Base.

Supports two dataset sources:
  1. amanuelbyte/acl6060-voice-cloning (existing dataset with reference_audio_path)
  2. ymoslem/acl-6060 (original ACL 60/60 with source audio)

This model supports:
  - 150+ languages (including Arabic!)
  - True voice cloning from a reference audio
  - Multi-speaker, multilingual generation

Usage:
    # Using the existing amanuelbyte dataset (default):
    python clone_with_scicom.py \\
        --output_dir ./output/acl6060_scicom \\
        --target_languages fr,zh,ar \\
        --max_rows 40

    # Using the original ymoslem/acl-6060 dataset:
    python clone_with_scicom.py \\
        --dataset ymoslem/acl-6060 \\
        --output_dir ./output/acl6060_scicom \\
        --target_languages fr,zh,ar \\
        --split dev

    # Quick test:
    python clone_with_scicom.py \\
        --output_dir ./output/acl6060_scicom \\
        --target_languages fr \\
        --max_rows 5

Requirements:
    pip install transformers torch soundfile librosa neucodec datasets pandas tqdm
"""

import argparse
import os
import re
import sys

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import load_dataset
from tqdm import tqdm


# ─── Constants ──────────────────────────────────────────────────────────────────

MODEL_NAME = "Scicom-intl/Multilingual-TTS-1.7B-Base"
OUTPUT_SAMPLE_RATE = 24000  # NeuCodec outputs 24kHz
CODEC_INPUT_SR = 16000      # NeuCodec expects 16kHz input

# ── Dataset configs ─────────────────────────────────────────────────────────

DATASET_CONFIGS = {
    "amanuelbyte/acl6060-voice-cloning": {
        "audio_col": "reference_audio_path",
        "source_text_col": "reference_text_en",
        "text_columns": {
            "en": "target_text_en",
            "ar": "target_text_ar",
            "fr": "target_text_fr",
        },
        "default_split": "train",
        "id_col": "sample_id",
        "speaker_col": "speaker_id",
    },
    "ymoslem/acl-6060": {
        "audio_col": "audio",
        "source_text_col": "text_en",
        "text_columns": {
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
        },
        "default_split": "dev",
        "id_col": "index",
        "speaker_col": None,
    },
}


# ─── Model Loading ─────────────────────────────────────────────────────────────

_codec = None
_model = None
_tokenizer = None


def load_models(device: str = "cuda"):
    """Load the Scicom TTS model, tokenizer, and NeuCodec."""
    global _codec, _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer, _codec

    from neucodec import NeuCodec
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading NeuCodec...")
    _codec = NeuCodec.from_pretrained("neuphonic/neucodec")
    _codec = _codec.eval().to(device)

    print(f"Loading model: {MODEL_NAME}...")
    _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    _model = _model.to(device)
    _model.eval()

    print(f"Loading tokenizer: {MODEL_NAME}...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("All models loaded successfully.")
    return _model, _tokenizer, _codec


# ─── Audio Encoding / Decoding ─────────────────────────────────────────────────


def encode_reference_audio(audio_array: np.ndarray, sr: int, codec, device: str = "cuda") -> str:
    """
    Encode a reference audio into codec tokens for voice cloning.

    Returns:
        String of codec tokens like '<|s_0|><|s_1|>...'
    """
    # Resample to 16kHz if needed
    if sr != CODEC_INPUT_SR:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=CODEC_INPUT_SR)

    audio_tensor = torch.tensor(audio_array, dtype=torch.float32)[None, None].to(device)

    with torch.no_grad():
        codes = codec.encode_code(audio_tensor)

    tokens = "".join([f"<|s_{i}|>" for i in codes[0, 0]])
    return tokens


def decode_audio_tokens(generated_text: str, codec, device: str = "cuda") -> np.ndarray | None:
    """
    Extract audio tokens from generated text and decode to waveform.

    Returns:
        Audio waveform as numpy array, or None if no tokens found.
    """
    parts = generated_text.split("<|speech_start|>")
    if len(parts) < 2:
        return None

    last_speech = parts[-1]
    audio_tokens = re.findall(r"<\|s_(\d+)\|>", last_speech)

    if not audio_tokens:
        return None

    audio_codes = torch.tensor([int(t) for t in audio_tokens])[None, None].to(device)

    with torch.no_grad():
        audio_waveform = codec.decode_code(audio_codes)

    return audio_waveform[0, 0].cpu().numpy()


# ─── Voice Cloning Generation ──────────────────────────────────────────────────


def generate_cloned_speech(
    model,
    tokenizer,
    reference_text: str,
    reference_tokens: str,
    target_text: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.8,
    repetition_penalty: float = 1.15,
) -> str:
    """
    Generate speech in a cloned voice.

    Uses the voice cloning prompt format:
      <|im_start|>{ref_text}<|speech_start|>{ref_tokens}<|im_end|>
      <|im_start|>{target_text}<|speech_start|>
    """
    prompt = (
        f"<|im_start|>{reference_text}<|speech_start|>{reference_tokens}<|im_end|>"
        f"<|im_start|>{target_text}<|speech_start|>"
    )

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=False)


# ─── Main Pipeline ─────────────────────────────────────────────────────────────


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def process_dataset(
    dataset_name: str,
    output_dir: str,
    target_languages: list[str],
    split: str | None = None,
    max_rows: int | None = None,
    device: str = "cuda",
    max_new_tokens: int = 2048,
    temperature: float = 0.8,
):
    """
    Process a dataset with true voice cloning.

    For each row:
    1. Encode source/reference audio -> codec tokens (captures speaker voice)
    2. Get translated text from existing dataset columns
    3. Generate cloned speech preserving original speaker's voice
    """
    config = DATASET_CONFIGS[dataset_name]
    use_split = split or config["default_split"]

    audio_dir = ensure_dir(os.path.join(output_dir, "cloned_audio"))

    # Load dataset
    print(f"Loading dataset: {dataset_name} (split={use_split})")
    ds = load_dataset(dataset_name, split=use_split)
    if max_rows and max_rows < len(ds):
        ds = ds.select(range(max_rows))
        print(f"  Selected first {max_rows} rows.")
    print(f"  Total rows: {len(ds)}")
    print(f"  Columns: {ds.column_names}")

    # Validate target languages
    available_langs = config["text_columns"]
    for lang in target_languages:
        if lang not in available_langs:
            print(
                f"  ⚠ Language '{lang}' not available in {dataset_name}. "
                f"Available: {list(available_langs.keys())}"
            )

    model, tokenizer, codec = load_models(device=device)

    records = []
    total = len(ds) * len(target_languages)
    pbar = tqdm(total=total, desc="Voice cloning", unit="clip")

    for idx in range(len(ds)):
        row = ds[idx]

        # ── Extract audio and encode speaker identity ───────────────────
        audio_data = row.get(config["audio_col"])
        if audio_data is None:
            print(f"\n  ⚠ Row {idx}: no audio found, skipping.")
            pbar.update(len(target_languages))
            continue

        audio_array = np.array(audio_data["array"], dtype=np.float32)
        sr = audio_data["sampling_rate"]
        source_text = row.get(config["source_text_col"], "")

        try:
            ref_tokens = encode_reference_audio(audio_array, sr, codec, device)
        except Exception as e:
            print(f"\n  ✗ Row {idx}: failed to encode reference audio: {e}")
            pbar.update(len(target_languages))
            continue

        # ── Build record ────────────────────────────────────────────────
        row_id = row.get(config["id_col"], idx)
        row_record = {
            "index": row_id,
            "source_text_en": source_text,
        }
        if config["speaker_col"] and config["speaker_col"] in ds.column_names:
            row_record["speaker_id"] = row.get(config["speaker_col"], "")

        # Copy existing text columns
        for lang_code, col_name in available_langs.items():
            if col_name in ds.column_names:
                row_record[f"text_{lang_code}"] = row.get(col_name, "")

        # ── Generate cloned speech per target language ──────────────────
        for target_lang in target_languages:
            lang_audio_dir = ensure_dir(os.path.join(audio_dir, target_lang))

            text_col = available_langs.get(target_lang)
            target_text = row.get(text_col, "") if text_col else ""

            if not target_text:
                print(f"\n  ⚠ Row {idx}: no text for '{target_lang}', skipping.")
                row_record[f"cloned_voice_{target_lang}"] = ""
                pbar.update(1)
                continue

            filename = f"cloned_{idx:05d}_{target_lang}.wav"
            filepath = os.path.join(lang_audio_dir, filename)

            try:
                generated_text = generate_cloned_speech(
                    model=model,
                    tokenizer=tokenizer,
                    reference_text=source_text,
                    reference_tokens=ref_tokens,
                    target_text=target_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )

                audio_waveform = decode_audio_tokens(generated_text, codec, device)

                if audio_waveform is not None and len(audio_waveform) > 0:
                    sf.write(filepath, audio_waveform, OUTPUT_SAMPLE_RATE)
                    row_record[f"cloned_voice_{target_lang}"] = os.path.relpath(
                        filepath, output_dir
                    )
                else:
                    print(f"\n  ⚠ Row {idx} -> {target_lang}: no audio tokens generated.")
                    row_record[f"cloned_voice_{target_lang}"] = ""

            except Exception as e:
                print(f"\n  ✗ Row {idx} -> {target_lang}: {e}")
                row_record[f"cloned_voice_{target_lang}"] = ""

            pbar.update(1)

        records.append(row_record)

    pbar.close()

    # ── Save metadata ───────────────────────────────────────────────────────
    if records:
        df = pd.DataFrame(records)
        csv_path = os.path.join(output_dir, "metadata_cloned.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved {len(records)} records to {csv_path}")

        json_path = os.path.join(output_dir, "metadata_cloned.json")
        df.to_json(json_path, orient="records", force_ascii=False, indent=2)
        print(f"✓ Saved metadata JSON to {json_path}")

        # Summary
        print("\n── Voice Cloning Summary ──")
        print(f"  Model:           {MODEL_NAME}")
        print(f"  Source dataset:   {dataset_name} ({use_split})")
        print(f"  Total rows:      {len(records)}")
        print(f"  Target languages: {target_languages}")
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
        description="Clone voices from ACL datasets using Scicom Multilingual-TTS-1.7B-Base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported datasets:
  amanuelbyte/acl6060-voice-cloning  (default, 40 rows, langs: en, ar, fr)
  ymoslem/acl-6060                   (468 rows, langs: en, ar, de, fa, fr, ja, nl, pt, ru, tr, zh)

This model supports 150+ languages including Arabic.
It performs TRUE voice cloning: the source speaker's voice is preserved.
        """,
    )
    parser.add_argument(
        "--dataset", type=str, default="amanuelbyte/acl6060-voice-cloning",
        choices=list(DATASET_CONFIGS.keys()),
        help="Source dataset to use.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/acl6060_scicom",
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--target_languages", type=str, default="fr,ar",
        help="Comma-separated target language codes.",
    )
    parser.add_argument(
        "--split", type=str, default=None,
        help="Dataset split (auto-detected if not provided).",
    )
    parser.add_argument(
        "--max_rows", type=int, default=None,
        help="Max rows to process (None = all).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"],
        help="Device for inference.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=2048,
        help="Max tokens to generate per sample.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    target_languages = [l.strip() for l in args.target_languages.split(",")]

    config = DATASET_CONFIGS[args.dataset]
    available = list(config["text_columns"].keys())

    invalid = [l for l in target_languages if l not in config["text_columns"]]
    if invalid:
        print(
            f"Error: Languages {invalid} not available in {args.dataset}. "
            f"Available: {available}"
        )
        sys.exit(1)

    print("=" * 60)
    print("  ACL Voice Cloning (Scicom Multilingual-TTS)")
    print("=" * 60)
    print(f"  Model:            {MODEL_NAME}")
    print(f"  Dataset:          {args.dataset}")
    print(f"  Split:            {args.split or config['default_split']}")
    print(f"  Target languages: {target_languages}")
    print(f"  Max rows:         {args.max_rows or 'all'}")
    print(f"  Output dir:       {args.output_dir}")
    print(f"  Device:           {args.device}")
    print(f"  Temperature:      {args.temperature}")
    print("=" * 60)

    process_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        target_languages=target_languages,
        split=args.split,
        max_rows=args.max_rows,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
