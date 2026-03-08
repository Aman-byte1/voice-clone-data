import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def fix_split(split, output_dir):
    print(f"\nFixing split {split}...")
    split_dir = os.path.join(output_dir, split)
    csv_path = os.path.join(split_dir, "metadata_cloned.csv")
    if not os.path.exists(csv_path):
        print(f"Skipping {split}, no metadata_cloned.csv found.")
        return

    df = pd.read_csv(csv_path)
    
    # Load original HF dataset to get the English audio
    print(f"  Fetching original HuggingFace dataset for {split}...")
    ds = load_dataset("amanuelbyte/acl6060-voice-cloning", split=split)
    
    audio_dir = ensure_dir(os.path.join(split_dir, "cloned_audio", "en"))
    
    new_records = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Reformatting {split}"):
        row_dict = row.to_dict()
        
        # Get original English audio from original dataset
        ds_row = ds[idx]
        audio_data = ds_row["reference_audio_path"]
        audio_array = np.array(audio_data["array"], dtype=np.float32)
        sr = audio_data["sampling_rate"]
        
        en_wav_path = os.path.join("cloned_audio", "en", f"original_{idx:05d}.wav")
        abs_en_wav_path = os.path.join(split_dir, en_wav_path)
        sf.write(abs_en_wav_path, audio_array, sr)
        
        # Build new record with requested column order
        final_record = {}
        
        # 1. En text and En audio
        final_record["text_en"] = str(row_dict.get("source_text_en", row_dict.get("text_en", "")))
        final_record["voice_en"] = en_wav_path
        
        # 2. Ar text and Ar audio
        final_record["text_ar"] = str(row_dict.get("text_ar", ""))
        ar_audio = row_dict.get("cloned_voice_ar", row_dict.get("voice_ar"))
        final_record["voice_ar"] = str(ar_audio) if pd.notna(ar_audio) else ""
        
        # 3. Fr text and Fr audio
        final_record["text_fr"] = str(row_dict.get("text_fr", ""))
        fr_audio = row_dict.get("cloned_voice_fr", row_dict.get("voice_fr"))
        final_record["voice_fr"] = str(fr_audio) if pd.notna(fr_audio) else ""
        
        # 4. Remaining columns
        for k, v in row_dict.items():
            if k not in ["source_text_en", "text_en", "text_ar", "text_fr", "cloned_voice_ar", "voice_ar", "cloned_voice_fr", "voice_fr"]:
                final_record[k] = v
                
        new_records.append(final_record)
        
    new_df = pd.DataFrame(new_records)
    new_df.to_csv(csv_path, index=False)
    print(f"✓ Saved updated nicely aligned metadata for {split}")

if __name__ == "__main__":
    output_dir = "./output/acl6060_scicom"
    for split in ["train", "test"]:
        fix_split(split, output_dir)
