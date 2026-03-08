import pandas as pd
from datasets import Dataset, Audio
import os

df = pd.DataFrame({
    "text": ["hello", "world"],
    "voice": ["nonexistent.wav", None]
})
data_dict = df.to_dict(orient="list")
ds = Dataset.from_dict(data_dict)
try:
    ds = ds.cast_column("voice", Audio())
    print("Cast successful. Features:", ds.features)
except Exception as e:
    print("Cast failed:", e)
