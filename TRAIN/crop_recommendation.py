import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# === Load Enhanced Dataset ===
data_path = r"D:\FINAL PROJECT\DATA\crops\Crop_recommendation_enhanced.csv"
df = pd.read_csv(data_path)

print("Enhanced dataset loaded successfully.")
print("Dataset shape:", df.shape)

# === Split Features & Target ===
X = df.drop("label", axis=1)
y = df["label"]

# === Split Data (Train/Test) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Train Random Forest Model ===
model = RandomForestClassifier(
    n_estimators=500,              # More trees for better accuracy
    max_depth=None,                # Let trees grow fully
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1                      # Use all CPU cores
)

print("\nTraining model, please wait...")
model.fit(X_train, y_train)
print("Training complete.")

# === Evaluate Model ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\nModel Accuracy: {accuracy:.2f}%")

# === Save Model ===
output_dir = r"D:\FINAL PROJECT\MODEL"
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "crop_recommendation_model.pkl")
joblib.dump(model, model_path)

print(f"Model saved successfully at: {model_path}")
