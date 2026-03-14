"""
Generate French cloned voice dataset with train/test splits.

Combines both splits of ymoslem/acl-6060 (884 total segments),
shuffles with seed 0, selects 100 for test and 784 for train,
then generates French cloned audio using Scicom Multilingual-TTS-1.7B-Base.

Usage (on server):
    python generate_french_dataset.py \
        --output_dir ./output/acl6060_fr \
        --device cuda
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
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

DATASET_NAME = "ymoslem/acl-6060"
TTS_MODEL_NAME = "Scicom-intl/Multilingual-TTS-1.7B-Base"
OUTPUT_SAMPLE_RATE = 24000  # NeuCodec outputs 24kHz
CODEC_INPUT_SR = 16000      # NeuCodec expects 16kHz input
NUM_TEST = 100
RANDOM_SEED = 0


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

    print(f"Loading model: {TTS_MODEL_NAME}...")
    _model = AutoModelForCausalLM.from_pretrained(TTS_MODEL_NAME)
    _model = _model.to(device)
    _model.eval()

    print(f"Loading tokenizer: {TTS_MODEL_NAME}...")
    _tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_NAME)

    print("All models loaded successfully.")
    return _model, _tokenizer, _codec


# ─── Audio Encoding / Decoding ─────────────────────────────────────────────────


def encode_reference_audio(audio_array, sr, codec, device="cuda"):
    """Encode a reference audio into codec tokens for voice cloning."""
    if sr != CODEC_INPUT_SR:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=CODEC_INPUT_SR)
    audio_tensor = torch.tensor(audio_array, dtype=torch.float32)[None, None].to(device)
    with torch.no_grad():
        codes = codec.encode_code(audio_tensor)
    tokens = "".join([f"<|s_{i}|>" for i in codes[0, 0]])
    return tokens


def decode_audio_tokens(generated_text, codec, device="cuda"):
    """Extract audio tokens from generated text and decode to waveform."""
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


def generate_cloned_speech(
    model, tokenizer, reference_text, reference_tokens, target_text,
    max_new_tokens=2048, temperature=0.8, repetition_penalty=1.15,
):
    """Generate speech in a cloned voice using Scicom prompt format."""
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


def generate_split(ds, split_name, output_dir, model, tokenizer, codec, device, temperature):
    """Generate French cloned audio for a single split using voice cloning."""
    split_dir = ensure_dir(os.path.join(output_dir, split_name))
    audio_dir = ensure_dir(os.path.join(split_dir, "cloned_audio_fr"))

    target_lang = "fr"
    records = []
    pbar = tqdm(total=len(ds), desc=f"Cloning French [{split_name}]", unit="clip")

    for idx in range(len(ds)):
        row = ds[idx]

        row_record = {
            "selected_id": idx,
            "original_index": row.get("index", ""),
            "source_text_en": row.get("text_en", ""),
            "text_fr": row.get("text_fr", ""),
            "tts_model": TTS_MODEL_NAME,
        }

        # Get source audio for voice cloning
        audio_val = row.get("audio")
        if audio_val is None or not isinstance(audio_val, dict):
            print(f"\n  ⚠ Row {idx}: no source audio, skipping.")
            row_record["cloned_voice_fr"] = ""
            records.append(row_record)
            pbar.update(1)
            continue

        audio_array = np.array(audio_val["array"], dtype=np.float32)
        sr = audio_val["sampling_rate"]
        source_text = row.get("text_en", "")
        translated_text = row.get("text_fr", "")

        if not translated_text:
            print(f"\n  ⚠ Row {idx}: no French text, skipping.")
            row_record["cloned_voice_fr"] = ""
            records.append(row_record)
            pbar.update(1)
            continue

        # Encode reference audio for voice cloning
        try:
            ref_tokens = encode_reference_audio(audio_array, sr, codec, device)
        except Exception as e:
            print(f"\n  ✗ Row {idx}: failed to encode reference: {e}")
            row_record["cloned_voice_fr"] = ""
            records.append(row_record)
            pbar.update(1)
            continue

        filename = f"cloned_{idx:05d}_fr.wav"
        filepath = os.path.join(audio_dir, filename)

        try:
            generated_text = generate_cloned_speech(
                model=model, tokenizer=tokenizer,
                reference_text=source_text, reference_tokens=ref_tokens,
                target_text=translated_text, temperature=temperature,
            )
            audio_waveform = decode_audio_tokens(generated_text, codec, device)

            if audio_waveform is not None and len(audio_waveform) > 0:
                sf.write(filepath, audio_waveform, OUTPUT_SAMPLE_RATE)
                row_record["cloned_voice_fr"] = os.path.relpath(filepath, split_dir)
            else:
                row_record["cloned_voice_fr"] = ""
                print(f"\n  ⚠ Row {idx}: no audio tokens generated")
        except Exception as e:
            print(f"\n  ✗ Failed row {idx} -> fr: {e}")
            row_record["cloned_voice_fr"] = ""

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
        description="Generate French cloned voice dataset (train + test) using Scicom TTS."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/acl6060_fr",
        help="Root output directory.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"],
        help="Device for inference.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature.",
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
    print(f"  Temperature:     {args.temperature}")
    print("=" * 60)

    ds_train, ds_test = load_and_split_dataset()
    model, tokenizer, codec = load_models(device=args.device)

    print("\n── Generating TEST split ──")
    generate_split(ds_test, "test", args.output_dir, model, tokenizer, codec, args.device, args.temperature)

    print("\n── Generating TRAIN split ──")
    generate_split(ds_train, "train", args.output_dir, model, tokenizer, codec, args.device, args.temperature)

    print("\n" + "=" * 60)
    print("  ✓ All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
