import pandas as pd
import numpy as np
from sklearn.utils import resample
import os

# === Load Original Dataset ===
input_path = r"F:\FINAL PROJECT\DATA\crops\Crop_recommendation.csv"
df = pd.read_csv(input_path)

print("Original dataset loaded successfully.")
print("Dataset shape:", df.shape)

# === 1. Balance the dataset (equal samples per crop) ===
max_count = df['label'].value_counts().max()
balanced_df = pd.DataFrame()

for crop in df['label'].unique():
    subset = df[df['label'] == crop]
    upsampled = resample(subset, replace=True, n_samples=max_count, random_state=42)
    balanced_df = pd.concat([balanced_df, upsampled])

df = balanced_df.reset_index(drop=True)

print("Dataset balanced.")

# === 2. Widen Weather Variability ===
df["temperature"] += np.random.uniform(-3, 3, len(df))
df["humidity"] += np.random.uniform(-8, 8, len(df))
df["rainfall"] += np.random.uniform(-15, 15, len(df))

# === 3. Emphasize Soil Nutrient Importance ===
df["N"] *= 1.2
df["P"] *= 1.1
df["K"] *= 1.1

# === 4. Add Seasonal Variation ===
seasons = ["kharif", "rabi", "summer"]
df["season"] = np.random.choice(seasons, len(df))

# Convert categorical season into numeric (one-hot encoding)
df = pd.get_dummies(df, columns=["season"])

# === 5. Add Gaussian Noise to Improve Generalization ===
numeric_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
df[numeric_cols] = df[numeric_cols] + np.random.normal(0, 0.02, df[numeric_cols].shape)

# === 6. Clip to Keep Realistic Values ===
df["N"] = df["N"].clip(0, 200)
df["P"] = df["P"].clip(0, 200)
df["K"] = df["K"].clip(0, 200)
df["temperature"] = df["temperature"].clip(10, 45)
df["humidity"] = df["humidity"].clip(20, 90)
df["ph"] = df["ph"].clip(4.5, 8.5)
df["rainfall"] = df["rainfall"].clip(0, 300)

# === 7. Save Enhanced Dataset ===
output_dir = r"F:\FINAL PROJECT\DATA\crops"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "Crop_recommendation_enhanced.csv")
df.to_csv(output_path, index=False)

print(f"Enhanced dataset saved successfully at: {output_path}")
