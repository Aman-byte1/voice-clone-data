
import sys
import os

# Add current directory to path
sys.path.append(r'c:\Users\geama\Documents\research\voice_clone_data')

from generate_french_dataset import load_and_split_dataset

def verify():
    print("Verifying dataset split...")
    ds_train, ds_test = load_and_split_dataset()
    
    print("\n--- TEST SET ---")
    print(f"Total rows: {len(ds_test)}")
    speakers_test = set(ds_test["speaker_id"])
    print(f"Speakers in test: {sorted(list(speakers_test))}")
    print(f"First 3 rows (keys): {ds_test['_resume_key'][:3]}")
    
    print("\n--- TRAIN SET ---")
    print(f"Total rows: {len(ds_train)}")
    speakers_train = set(ds_train["speaker_id"])
    print(f"Speakers in train: {sorted(list(speakers_train))}")
    print(f"First 3 rows (keys): {ds_train['_resume_key'][:3]}")
    
    # Check if Elena and Antoine are ONLY in test
    elena_antoine = {"speaker_01", "speaker_02"}
    if elena_antoine.issubset(speakers_test) and not elena_antoine.intersection(speakers_train):
        print("\n✓ SUCCESS: Elena and Antoine are exclusively in the test set.")
    else:
        print("\n✗ FAILURE: Speaker split is incorrect.")

    # Check order (should be sequential keys if possible to verify)
    # dev:0, dev:1, dev:2...
    first_test_key = ds_test['_resume_key'][0]
    first_train_key = ds_train['_resume_key'][0]
    print(f"\nFirst test key: {first_test_key}")
    print(f"First train key: {first_train_key}")

if __name__ == "__main__":
    verify()
