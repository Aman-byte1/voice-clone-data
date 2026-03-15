from datasets import load_dataset
DATASET_NAME = "ymoslem/acl-6060"
print(f"Checking {DATASET_NAME}...")
try:
    ds = load_dataset(DATASET_NAME, split="dev", streaming=True)
    row = next(iter(ds))
    print("Features:", ds.features)
    print("Keys in row:", row.keys())
    print("Audio content type:", type(row.get("audio")))
    print("Audio content:", row.get("audio"))
except Exception as e:
    print("Error:", e)
