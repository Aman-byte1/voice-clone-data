"""
Clone voices from the ACL 60/60 dataset using Scicom Multilingual-TTS-1.7B-Base.

This model supports:
  - 150+ languages (including Arabic!)
  - True voice cloning from a reference audio
  - Multi-speaker, multilingual generation

For each row in the ACL 60/60 dataset, this script:
1. Encodes the source audio into codec tokens (speaker identity)
2. Uses the translated text + speaker tokens to generate cloned speech
   in each target language
3. Saves audio files and metadata

Usage:
    python clone_with_scicom.py \\
        --output_dir ./output/acl6060_scicom \\
        --target_languages fr,zh,ar \\
        --split dev \\
        --max_rows 468

    # Quick test on first 5 rows:
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
DATASET_NAME = "ymoslem/acl-6060"
OUTPUT_SAMPLE_RATE = 24000  # NeuCodec outputs 24kHz
CODEC_INPUT_SR = 16000      # NeuCodec expects 16kHz input

# Text columns in the ACL 60/60 dataset
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

    Args:
        audio_array: Audio waveform as numpy array.
        sr: Sample rate of the audio.
        codec: Loaded NeuCodec instance.
        device: Device string.

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

    Args:
        generated_text: Full generated text from the model.
        codec: Loaded NeuCodec instance.
        device: Device string.

    Returns:
        Audio waveform as numpy array, or None if no tokens found.
    """
    # Extract the last speech segment's tokens
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

    Args:
        model: Loaded CausalLM model.
        tokenizer: Loaded tokenizer.
        reference_text: Text spoken in the reference audio.
        reference_tokens: Codec tokens from the reference audio.
        target_text: Text to synthesize in the cloned voice.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        repetition_penalty: Repetition penalty.

    Returns:
        Full generated text including audio tokens.
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
    """Create directory if it doesn't exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


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


def process_dataset(
    output_dir: str,
    target_languages: list[str],
    split: str = "dev",
    max_rows: int | None = None,
    device: str = "cuda",
    max_new_tokens: int = 2048,
    temperature: float = 0.8,
):
    """
    Process the ACL 60/60 dataset with true voice cloning.

    For each row:
    1. Encode source audio -> codec tokens (captures speaker voice)
    2. For each target language, generate cloned speech using the
       reference voice + translated text
    """
    audio_dir = ensure_dir(os.path.join(output_dir, "cloned_audio"))
    ds = load_acl_dataset(split=split, max_rows=max_rows)
    model, tokenizer, codec = load_models(device=device)

    records = []
    total = len(ds) * len(target_languages)
    pbar = tqdm(total=total, desc="Voice cloning", unit="clip")

    for idx in range(len(ds)):
        row = ds[idx]

        # ── Extract source audio and encode speaker identity ────────────
        audio_data = row.get("audio")
        if audio_data is None:
            print(f"\n  ⚠ Row {idx}: no audio found, skipping.")
            pbar.update(len(target_languages))
            continue

        audio_array = np.array(audio_data["array"], dtype=np.float32)
        sr = audio_data["sampling_rate"]
        source_text = row.get("text_en", "")

        # Encode reference audio into codec tokens (speaker identity)
        try:
            ref_tokens = encode_reference_audio(audio_array, sr, codec, device)
        except Exception as e:
            print(f"\n  ✗ Row {idx}: failed to encode reference audio: {e}")
            pbar.update(len(target_languages))
            continue

        # ── Build base record ───────────────────────────────────────────
        row_record = {
            "index": row.get("index", idx),
            "source_text_en": source_text,
        }

        # Copy all existing text translations
        for lang_code, col_name in ACL_TEXT_COLUMNS.items():
            if col_name in ds.column_names:
                row_record[f"text_{lang_code}"] = row.get(col_name, "")

        # ── Generate cloned speech for each target language ─────────────
        for target_lang in target_languages:
            lang_audio_dir = ensure_dir(os.path.join(audio_dir, target_lang))

            text_col = ACL_TEXT_COLUMNS.get(target_lang)
            target_text = row.get(text_col, "") if text_col else ""

            if not target_text:
                print(f"\n  ⚠ Row {idx}: no text for '{target_lang}', skipping.")
                row_record[f"cloned_voice_{target_lang}"] = ""
                pbar.update(1)
                continue

            filename = f"cloned_{idx:05d}_{target_lang}.wav"
            filepath = os.path.join(lang_audio_dir, filename)

            try:
                # Generate cloned speech
                generated_text = generate_cloned_speech(
                    model=model,
                    tokenizer=tokenizer,
                    reference_text=source_text,
                    reference_tokens=ref_tokens,
                    target_text=target_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )

                # Decode audio tokens to waveform
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
        print(f"  Source dataset:   {DATASET_NAME} ({split})")
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
        description="Clone voices from ACL 60/60 using Scicom Multilingual-TTS-1.7B-Base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This model supports 150+ languages including Arabic.
It performs TRUE voice cloning: the source speaker's voice is preserved
in the generated target-language speech.

ACL 60/60 text columns: en, ar, de, fa, fr, ja, nl, pt, ru, tr, zh
        """,
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/acl6060_scicom",
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--target_languages", type=str, default="fr,zh,ar",
        help="Comma-separated target language codes.",
    )
    parser.add_argument(
        "--split", type=str, default="dev",
        help="Dataset split (dev or test).",
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

    # Validate languages exist in ACL dataset
    for lang in target_languages:
        if lang not in ACL_TEXT_COLUMNS:
            print(
                f"Error: Language '{lang}' not in ACL 60/60 dataset. "
                f"Available: {list(ACL_TEXT_COLUMNS.keys())}"
            )
            sys.exit(1)

    print("=" * 60)
    print("  ACL 60/60 Voice Cloning (Scicom Multilingual-TTS)")
    print("=" * 60)
    print(f"  Model:            {MODEL_NAME}")
    print(f"  Dataset:          {DATASET_NAME}")
    print(f"  Split:            {args.split}")
    print(f"  Target languages: {target_languages}")
    print(f"  Max rows:         {args.max_rows or 'all'}")
    print(f"  Output dir:       {args.output_dir}")
    print(f"  Device:           {args.device}")
    print(f"  Temperature:      {args.temperature}")
    print("=" * 60)

    process_dataset(
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
