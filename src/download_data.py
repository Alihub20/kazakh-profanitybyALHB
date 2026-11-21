# src/download_data.py
# Downloads KazSAnDRA and saves a random sample of 2000 reviews.
import pandas as pd
from datasets import load_dataset

SEED = 42
SAMPLE_SIZE = 2000

# 1. Download the dataset (all splits) from Hugging Face
ds = load_dataset("issai/kazsandra", "full")
df = pd.concat([ds[split].to_pandas() for split in ds], ignore_index=True)
print("Columns:", list(df.columns))
print("Total reviews:", len(df))

# 2. Find the column that contains the review text
text_col = "text" if "text" in df.columns else df.select_dtypes("object").columns[0]
df = df.rename(columns={text_col: "text"})

# 3. Clean: remove empty and duplicate reviews
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].drop_duplicates(subset="text")
print("After removing duplicates:", len(df))

# 4. Save ALL reviews (used in Step 3 to search for swear words)
df.to_csv("data/all_reviews.csv", index=False, encoding="utf-8")

# 5. Save a random sample of 2000 reviews
sample = df.sample(n=SAMPLE_SIZE, random_state=SEED)
sample.to_csv("data/raw_reviews.csv", index=False, encoding="utf-8")
print("Saved", len(sample), "reviews to data/raw_reviews.csv")