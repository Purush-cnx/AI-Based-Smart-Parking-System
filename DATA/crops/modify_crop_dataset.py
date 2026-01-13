# ==============================================
# Crop Recommendation Dataset Modification Script (No District Column)
# Author: Purushotham K
# ==============================================

import pandas as pd
import numpy as np
from sklearn.utils import resample
import os

# === Load Original Dataset ===
input_path = r"F:\FINAL PROJECT\DATA\crops\Crop_recommendation.csv"
df = pd.read_csv(input_path)

print("✅ Original dataset loaded successfully!")
print("Original shape:", df.shape)
print("\nCrop distribution before balancing:\n")
print(df['label'].value_counts())

# === 1️⃣ Balance the dataset (equal samples per crop) ===
max_count = df['label'].value_counts().max()
balanced_df = pd.DataFrame()

for crop in df['label'].unique():
    subset = df[df['label'] == crop]
    upsampled = resample(subset, replace=True, n_samples=max_count, random_state=42)
    balanced_df = pd.concat([balanced_df, upsampled])

df = balanced_df.reset_index(drop=True)

print("\n✅ Dataset balanced successfully!")
print("New shape:", df.shape)
print(df['label'].value_counts().head())

# === 2️⃣ Add slight random variation (noise) to make it more realistic ===
def add_noise(series, pct=0.05):
    """Add random ±pct% variation to a numeric series"""
    return series + np.random.uniform(-pct, pct, size=len(series)) * series

for col in ["N", "P", "K", "temperature", "humidity", "rainfall"]:
    df[col] = add_noise(df[col], 0.1)  # ±10% variation

# === 3️⃣ Clip values to realistic agricultural ranges ===
df["N"] = df["N"].clip(0, 200)
df["P"] = df["P"].clip(0, 200)
df["K"] = df["K"].clip(0, 200)
df["temperature"] = df["temperature"].clip(10, 45)
df["humidity"] = df["humidity"].clip(20, 90)
df["ph"] = df["ph"].clip(4.5, 8.5)
df["rainfall"] = df["rainfall"].clip(0, 300)

# === 4️⃣ Save the modified dataset ===
output_dir = r"F:\FINAL PROJECT\DATA\crops"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "Crop_recommendation_modified.csv")

df.to_csv(output_path, index=False)
print(f"\n✅ Modified dataset saved successfully at:\n{output_path}")

# === 5️⃣ Preview few samples ===
print("\n🔍 Sample data preview:")
print(df.head(10))
