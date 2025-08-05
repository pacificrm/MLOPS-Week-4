import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random

# Parse command-line argument
parser = argparse.ArgumentParser(description="Poison iris training labels.")
parser.add_argument("percent", type=float, help="Percentage of labels to poison (e.g., 5 for 5%)")
args = parser.parse_args()

# Load training data
df = pd.read_csv("data/train_iris.csv")

# Encode labels
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["species"])

# Calculate number of labels to poison
num_samples = len(df)
num_to_poison = int((args.percent / 100.0) * num_samples)
print(f"Poisoning {num_to_poison} out of {num_samples} samples...")

# Get all unique label values
unique_labels = list(df["label_encoded"].unique())

# Pick indices to poison
indices_to_poison = random.sample(range(num_samples), num_to_poison)

# Switch labels randomly (to a different class)
for idx in indices_to_poison:
    original = df.loc[idx, "label_encoded"]
    new_label = random.choice([l for l in unique_labels if l != original])
    df.at[idx, "label_encoded"] = new_label

# Replace poisoned encoded labels with actual species names
df["species"] = le.inverse_transform(df["label_encoded"])

# Drop temporary column
df = df.drop(columns=["label_encoded"])

# Save poisoned dataset
df.to_csv("data/iris_train_poison.csv", index=False)
print("✅ Saved poisoned dataset as data/iris_train_poison.csv")
