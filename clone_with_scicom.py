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


def process_split(
    dataset_name: str,
    output_dir: str,
    target_languages: list[str],
    split: str,
    max_rows: int | None,
    device: str,
    max_new_tokens: int,
    temperature: float,
    model,
    tokenizer,
    codec,
):
    """Process a single split."""
    config = DATASET_CONFIGS[dataset_name]
    split_dir = ensure_dir(os.path.join(output_dir, split))
    audio_dir = ensure_dir(os.path.join(split_dir, "cloned_audio"))

    print(f"\n========================================")
    print(f" Processing split: {split}")
    print(f"========================================")
    
    ds = load_dataset(dataset_name, split=split)
    if max_rows and max_rows < len(ds):
        ds = ds.select(range(max_rows))
        print(f"  Selected first {max_rows} rows.")
    print(f"  Total rows: {len(ds)}")

    available_langs = config["text_columns"]
    records = []
    total = len(ds) * len(target_languages)
    pbar = tqdm(total=total, desc=f"Cloning {split}", unit="clip")

    for idx in range(len(ds)):
        row = ds[idx]
        audio_data = row.get(config["audio_col"])
        if audio_data is None:
            pbar.update(len(target_languages))
            continue

        audio_array = np.array(audio_data["array"], dtype=np.float32)
        sr = audio_data["sampling_rate"]
        source_text = row.get(config["source_text_col"], "")

        # Get the original english audio path if it exists
        original_audio_path = row.get("reference_audio_path", "")
        # Fallback to saving the array as wav if "reference_audio_path" is missing
        if not original_audio_path:
            en_audio_dir = ensure_dir(os.path.join(audio_dir, "en"))
            original_audio_path = os.path.join(en_audio_dir, f"original_{idx:05d}.wav")
            sf.write(original_audio_path, audio_array, sr)

        try:
            ref_tokens = encode_reference_audio(audio_array, sr, codec, device)
        except Exception as e:
            print(f"\n  ✗ Row {idx}: failed to encode reference audio: {e}")
            pbar.update(len(target_languages))
            continue

        # ── Start building structured record ──
        row_id = row.get(config["id_col"], idx)
        temp_record = {
            "index": row_id,
            "speaker_id": row.get(config["speaker_col"], "") if config["speaker_col"] else "",
        }

        # Keep original text columns
        for lang_code, col_name in available_langs.items():
            if col_name in ds.column_names:
                temp_record[f"text_{lang_code}"] = row.get(col_name, "")
        temp_record["text_en"] = source_text # Ensure fallback mapping

        # Ensure en audio exists in the dataset output
        temp_record["cloned_voice_en"] = original_audio_path

        for target_lang in target_languages:
            lang_audio_dir = ensure_dir(os.path.join(audio_dir, target_lang))
            text_col = available_langs.get(target_lang)
            target_text = row.get(text_col, "") if text_col else ""

            if not target_text:
                temp_record[f"cloned_voice_{target_lang}"] = ""
                pbar.update(1)
                continue

            filename = f"cloned_{idx:05d}_{target_lang}.wav"
            filepath = os.path.join(lang_audio_dir, filename)

            try:
                generated_text = generate_cloned_speech(
                    model=model, tokenizer=tokenizer,
                    reference_text=source_text, reference_tokens=ref_tokens,
                    target_text=target_text, max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                audio_waveform = decode_audio_tokens(generated_text, codec, device)
                if audio_waveform is not None and len(audio_waveform) > 0:
                    sf.write(filepath, audio_waveform, OUTPUT_SAMPLE_RATE)
                    temp_record[f"cloned_voice_{target_lang}"] = os.path.relpath(filepath, split_dir)
                else:
                    temp_record[f"cloned_voice_{target_lang}"] = ""
            except Exception as e:
                print(f"\n  ✗ Row {idx} -> {target_lang}: {e}")
                temp_record[f"cloned_voice_{target_lang}"] = ""

            pbar.update(1)

        # Build final record with specific column order
        final_record = {}
        
        # 1. En text and En audio
        final_record["text_en"] = temp_record.get("text_en", "")
        final_record["cloned_voice_en"] = temp_record.get("cloned_voice_en", "")
        
        # 2. Ar text and Ar audio (if available)
        final_record["text_ar"] = temp_record.get("text_ar", "")
        final_record["cloned_voice_ar"] = temp_record.get("cloned_voice_ar", "")
        
        # 3. Fr text and Fr audio (if available)
        final_record["text_fr"] = temp_record.get("text_fr", "")
        final_record["cloned_voice_fr"] = temp_record.get("cloned_voice_fr", "")

        # 4. Remaining columns
        for k, v in row.items():
            if k not in ["reference_audio_path", "audio", "target_text_en", "target_text_ar", "target_text_fr", "reference_text_en", "text_en", "text_ar", "text_fr"]:
                final_record[k] = v

        records.append(final_record)
    pbar.close()

    if records:
        df = pd.DataFrame(records)
        csv_path = os.path.join(split_dir, "metadata_cloned.csv")
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved {len(records)} records for {split} to {csv_path}")


def process_dataset(
    dataset_name: str,
    output_dir: str,
    target_languages: list[str],
    splits: list[str] | None = None,
    max_rows: int | None = None,
    device: str = "cuda",
    max_new_tokens: int = 2048,
    temperature: float = 0.8,
):
    """Process multiple splits with true voice cloning."""
    config = DATASET_CONFIGS[dataset_name]
    use_splits = splits if splits else [config["default_split"]]

    model, tokenizer, codec = load_models(device=device)

    for split in use_splits:
        process_split(
            dataset_name=dataset_name,
            output_dir=output_dir,
            target_languages=target_languages,
            split=split,
            max_rows=max_rows,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            model=model,
            tokenizer=tokenizer,
            codec=codec,
        )

    print("\n── Multi-Split Voice Cloning Complete ──")
    print(f"  Output dir: {os.path.abspath(output_dir)}")
    print(f"  Splits processed: {use_splits}")


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
        "--splits", type=str, default=None,
        help="Comma-separated list of dataset splits (e.g., 'train,test'). Auto-detects if None.",
    )
    parser.add_argument(
        "--max_rows", type=int, default=None,
        help="Max rows to process per split (None = all).",
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
    splits = [s.strip() for s in args.splits.split(",")] if args.splits else None

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
    print(f"  Splits:           {splits or config['default_split']}")
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
        splits=splits,
        max_rows=args.max_rows,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
