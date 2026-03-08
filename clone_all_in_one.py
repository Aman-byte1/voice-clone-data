import os
import re
import torch
import numpy as np
import soundfile as sf
import librosa
import traceback
from datasets import load_dataset, Audio, DatasetDict
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from neucodec import NeuCodec

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Please set the HF_TOKEN environment variable.")
    
HF_USERNAME = "amanuelbyte"
DATASET_NAME = "amanuelbyte/acl6060-voice-cloning"
OUTPUT_DATASET_NAME = "acl6060-voice-cloning-multilingual"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLITS = ["train", "test"]

# ──────────────────────────────────────────────
# COLUMN NAMES FROM ORIGINAL DATASET
# ──────────────────────────────────────────────
REFERENCE_AUDIO_COL = "reference_audio_path"
EN_TEXT_COL = "reference_text_en"
AR_TEXT_COL = "target_text_ar"
FR_TEXT_COL = "target_text_fr"


# ──────────────────────────────────────────────
# LOGIN & LOAD MODELS
# ──────────────────────────────────────────────
login(token=HF_TOKEN)
print("✅ Logged in to HuggingFace")

print("⏳ Loading NeuCodec...")
codec = NeuCodec.from_pretrained("neuphonic/neucodec")
codec = codec.eval().to(DEVICE)
print("✅ NeuCodec loaded")

print("⏳ Loading Multilingual-TTS-1.7B-Base...")
model = AutoModelForCausalLM.from_pretrained(
    "Scicom-intl/Multilingual-TTS-1.7B-Base",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Scicom-intl/Multilingual-TTS-1.7B-Base")
print(f"✅ TTS model loaded on {DEVICE}")


# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────
def encode_reference_audio(audio_entry):
    array = np.array(audio_entry["array"], dtype=np.float32)
    sr = audio_entry["sampling_rate"]

    if sr != 16000:
        array = librosa.resample(array, orig_sr=sr, target_sr=16000)

    audio_tensor = torch.tensor(array, dtype=torch.float32)[None, None].to(DEVICE)
    with torch.no_grad():
        codes = codec.encode_code(audio_tensor)

    tokens = ''.join([f'<|s_{i}|>' for i in codes[0, 0]])
    return tokens


def generate_cloned_speech(reference_audio_entry, reference_text, target_text):
    if not target_text or not str(target_text).strip():
        return None

    try:
        ref_tokens = encode_reference_audio(reference_audio_entry)
        prompt = (
            f"<|im_start|>{reference_text}<|speech_start|>"
            f"{ref_tokens}<|im_end|>"
            f"<|im_start|>{target_text}<|speech_start|>"
        )

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)

        input_len = inputs["input_ids"].shape[1]
        if input_len > 8000:
            print(f"      ⚠️ Input too long ({input_len} tokens), trimming reference audio")
            short_audio = {
                "array": reference_audio_entry["array"][:16000 * 3],
                "sampling_rate": reference_audio_entry["sampling_rate"]
            }
            ref_tokens = encode_reference_audio(short_audio)
            prompt = (
                f"<|im_start|>{reference_text}<|speech_start|>"
                f"{ref_tokens}<|im_end|>"
                f"<|im_start|>{target_text}<|speech_start|>"
            )
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.8,
                repetition_penalty=1.15,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        audio_tokens = re.findall(r'<\|s_(\d+)\|>', generated_text.split('<|speech_start|>')[-1])

        if not audio_tokens:
            print("      ⚠️ No audio tokens generated")
            return None

        audio_tokens = [int(token) for token in audio_tokens]
        audio_codes = torch.tensor(audio_tokens)[None, None].to(DEVICE)

        with torch.no_grad():
            audio_waveform = codec.decode_code(audio_codes)

        audio_np = audio_waveform[0, 0].cpu().numpy().astype(np.float32)

        return {"array": audio_np, "sampling_rate": 24000}

    except Exception as e:
        print(f"      ⚠️ Error: {e}")
        traceback.print_exc()
        return None

# ──────────────────────────────────────────────
# PROCESS DATASET
# ──────────────────────────────────────────────
final_dataset_dict = DatasetDict()

for split in SPLITS:
    print(f"\n========================================")
    print(f" Processing split: {split}")
    print(f"========================================")
    
    ds = load_dataset(DATASET_NAME, split=split)
    print(f"✅ Loaded {len(ds)} rows for {split}")

    fr_cloned_audios = []
    ar_cloned_audios = []

    for i in range(len(ds)):
        row = ds[i]
        print(f"[{i+1}/{len(ds)}] Processing...")

        ref_audio = row.get(REFERENCE_AUDIO_COL)
        en_text = row.get(EN_TEXT_COL, "")
        fr_text = row.get(FR_TEXT_COL, "")
        ar_text = row.get(AR_TEXT_COL, "")

        if ref_audio is None:
            print(f"   ⚠️ No reference audio, skipping")
            fr_cloned_audios.append(None)
            ar_cloned_audios.append(None)
            continue

        # ── French cloned voice ──
        if fr_text and str(fr_text).strip():
            print(f"   🇫🇷 French: '{str(fr_text)[:60]}...'")
            fr_audio = generate_cloned_speech(ref_audio, en_text, fr_text)
            fr_cloned_audios.append(fr_audio)
            if fr_audio:
                print(f"      ✅ Generated {len(fr_audio['array'])/24000:.1f}s audio")
        else:
            fr_cloned_audios.append(None)

        # ── Arabic cloned voice ──
        if ar_text and str(ar_text).strip():
            print(f"   🇸🇦 Arabic: '{str(ar_text)[:60]}...'")
            ar_audio = generate_cloned_speech(ref_audio, en_text, ar_text)
            ar_cloned_audios.append(ar_audio)
            if ar_audio:
                print(f"      ✅ Generated {len(ar_audio['array'])/24000:.1f}s audio")
        else:
            ar_cloned_audios.append(None)

        if (i + 1) % 10 == 0:
            print(f"   📊 Progress: {i+1}/{len(ds)} rows done")
        if (i + 1) % 25 == 0:
            torch.cuda.empty_cache()

    # ──────────────────────────────────────────────
    # ADD COLUMNS & REORGANIZE
    # ──────────────────────────────────────────────
    print(f"⏳ Finalizing data format for {split}...")

    EMPTY_AUDIO = {"array": np.zeros(1, dtype=np.float32), "sampling_rate": 24000}
    
    # 1. Add safely padded audio columns
    ds = ds.add_column("voice_fr", [a if a is not None else EMPTY_AUDIO for a in fr_cloned_audios])
    ds = ds.add_column("voice_ar", [a if a is not None else EMPTY_AUDIO for a in ar_cloned_audios])

    # 2. Rename existing columns to perfectly match requested schema
    # Make sure we don't accidentally rename a column that's already named what we want.
    rename_mapping = {
        EN_TEXT_COL: "text_en",
        REFERENCE_AUDIO_COL: "voice_en",
        AR_TEXT_COL: "text_ar",
        FR_TEXT_COL: "text_fr"
    }

    for old_col, new_col in rename_mapping.items():
        if old_col in ds.column_names and old_col != new_col:
            if new_col in ds.column_names:
                 ds = ds.remove_columns([new_col])
            ds = ds.rename_column(old_col, new_col)

    # 3. Form select list to match exact requested ordering
    ordered_cols = ["text_en", "voice_en", "text_ar", "voice_ar", "text_fr", "voice_fr"]
    remaining_cols = [c for c in ds.column_names if c not in ordered_cols]
    
    # We select exactly the subset needed
    ds = ds.select_columns(ordered_cols + remaining_cols)

    # 4. Cast Audio features (voice_en is already Audio if loaded from Hub, but ensure Ar/Fr)
    ds = ds.cast_column("voice_fr", Audio(sampling_rate=24000))
    ds = ds.cast_column("voice_ar", Audio(sampling_rate=24000))
    # Note: voice_en uses whatever sampling_rate it originally had

    print(f"✅ {split} split ready. Columns: {ds.column_names}")
    final_dataset_dict[split] = ds

# ──────────────────────────────────────────────
# PUSH
# ──────────────────────────────────────────────
full_repo = f"{HF_USERNAME}/{OUTPUT_DATASET_NAME}"
print(f"\n⏳ Pushing exactly sorted multilingual dataset to {full_repo}...")

final_dataset_dict.push_to_hub(full_repo, token=HF_TOKEN, private=False)

print(f"✅ All Done! 🔗 https://huggingface.co/datasets/{full_repo}")
