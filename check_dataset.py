from datasets import load_dataset
DATASET_NAME = "ymoslem/acl-6060"
ds_dev = load_dataset(DATASET_NAME, split="dev")
print("Features:", ds_dev.features)
print("First row 'audio':", ds_dev[0].get("audio"))
print("First row 'text_fr':", ds_dev[0].get("text_fr"))
