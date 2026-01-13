import requests
import pandas as pd
import joblib
import numpy as np
import datetime

# === Weather API Key ===
API_KEY = "1259050f0b8c4711bf2112234250610"

# === Load Trained Model ===
MODEL_PATH = r"D:\FINAL PROJECT\MODEL\crop_recommendation_model.pkl"
model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

# === User Inputs ===
try:
    N = float(input("Enter Nitrogen (N): "))
    P = float(input("Enter Phosphorus (P): "))
    K = float(input("Enter Potassium (K): "))
    ph = float(input("Enter Soil pH: "))
    district = input("Enter District/City Name (e.g., Bangalore): ").strip()
except Exception:
    print("Invalid input! Please enter numeric values for N, P, K, and pH.")
    exit()

# === Fetch Live Weather Data ===
print("\nFetching live weather data...")
weather_url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={district}&aqi=no"

try:
    response = requests.get(weather_url)
    data = response.json()

    if "current" not in data:
        raise ValueError("Weather data not found for this location.")

    temperature = data["current"]["temp_c"]
    humidity = data["current"]["humidity"]
    rainfall = data["current"].get("precip_mm", 0)

    print("\nLive Weather Data:")
    print(f"Temperature : {temperature:.1f} °C")
    print(f"Humidity     : {humidity:.1f} %")
    print(f"Rainfall     : {rainfall:.1f} mm")

except Exception:
    print("\nFailed to fetch live weather data. Please enter manually.")
    temperature = float(input("Enter Temperature (°C): "))
    humidity = float(input("Enter Humidity (%): "))
    rainfall = float(input("Enter Rainfall (mm): "))

# === Determine Current Season Automatically ===
month = datetime.datetime.now().month
if month in [6, 7, 8, 9, 10]:
    season = "kharif"
elif month in [11, 12, 1, 2]:
    season = "rabi"
else:
    season = "summer"

# One-hot encode the season (matches training dataset)
season_cols = {"season_kharif": 0, "season_rabi": 0, "season_summer": 0}
season_cols[f"season_{season}"] = 1

print(f"\nCurrent detected season: {season.upper()}")

# === Prepare Input Data ===
input_data = pd.DataFrame([{
    "N": N * 1.2,
    "P": P * 1.1,
    "K": K * 1.1,
    "temperature": temperature,
    "humidity": humidity,
    "ph": ph,
    "rainfall": rainfall,
    **season_cols
}])

# Ensure all expected columns exist (fill missing with 0)
model_features = model.feature_names_in_
for col in model_features:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[model_features]

# === Make Prediction ===
predicted_crop = model.predict(input_data)[0]

print("\nRecommended Crop:")
print(f"=> {predicted_crop.upper()}")

# === Display Top 5 Probable Crops ===
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(input_data)[0]
    prob_df = pd.DataFrame({
        "Crop": model.classes_,
        "Confidence (%)": probabilities * 100
    }).sort_values(by="Confidence (%)", ascending=False)

    print("\nTop 5 Suitable Crops:")
    print(prob_df.head(5).to_string(index=False, formatters={'Confidence (%)': '{:.2f}'.format}))

print("\nPrediction completed successfully.")
