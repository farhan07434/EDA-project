import pandas as pd

# Load datasets
credits = pd.read_csv("credits.csv")
titles = pd.read_csv("titles.csv")

# Merge using common column 'id'
merged = pd.merge(titles, credits, on="id")

# Save new dataset
merged.to_csv("merged_dataset.csv", index=False)

print("Datasets merged successfully!")