# Fertilizer Recommendation System - Based on NPK Levels and Crop Type

import pandas as pd
import numpy as np
import joblib

# Sample fertilizer dictionary (can be expanded based on real data)
fertilizer_dict = {
    "NHigh": "Urea",
    "NLow": "Ammonium Sulphate",
    "PHigh": "Single Super Phosphate",
    "PLow": "DAP",
    "KHigh": "Muriate of Potash",
    "KLow": "Sulphate of Potash"
}

# Ideal NPK values for different crops (can be fine-tuned)
ideal_npk = {
    # Cereals
    "rice": [90, 40, 40],
    "wheat": [120, 60, 40],
    "maize": [120, 60, 40],
    "sorghum": [100, 50, 40],
    "barley": [80, 40, 40],

    # Pulses
    "chickpea": [20, 40, 20],
    "pigeonpea": [20, 50, 20],
    "green gram": [20, 40, 20],
    "black gram": [20, 40, 20],
    "lentil": [20, 40, 20],

    # Oilseeds
    "groundnut": [25, 50, 75],
    "mustard": [80, 40, 40],
    "sunflower": [60, 60, 40],
    "soybean": [30, 60, 40],
    "sesame": [40, 20, 20],

    # Commercial crops
    "cotton": [150, 75, 75],
    "sugarcane": [250, 115, 115],
    "jute": [60, 30, 30],
    "tobacco": [90, 60, 60],

    # Vegetables
    "tomato": [100, 60, 50],
    "potato": [150, 60, 120],
    "onion": [100, 50, 50],
    "brinjal": [100, 60, 50],
    "cabbage": [120, 60, 60],
    "cauliflower": [120, 60, 60],
    "chilli": [100, 50, 50],
    "okra": [80, 40, 40],
    "carrot": [60, 40, 40],

    # Fruits
    "banana": [200, 60, 200],
    "mango": [100, 50, 100],
    "grapes": [120, 60, 120],
    "orange": [120, 60, 120],
    "apple": [70, 35, 70]
}

# ==== Farmer Inputs ====
crop_name = input("Enter crop name: ").strip().lower()
N = int(input("Enter Nitrogen level (N): "))
P = int(input("Enter Phosphorus level (P): "))
K = int(input("Enter Potassium level (K): "))

# Check if crop is in the ideal NPK dictionary
if crop_name not in ideal_npk:
    print("Sorry, crop not found in our NPK database.")
    exit()

ideal_N, ideal_P, ideal_K = ideal_npk[crop_name]

# Check each nutrient level
suggestions = []

if N < ideal_N:
    suggestions.append(f"Nitrogen is low. Use {fertilizer_dict['NLow']}.")
elif N > ideal_N:
    suggestions.append(f"Nitrogen is high. Use {fertilizer_dict['NHigh']} cautiously or reduce its usage.")

if P < ideal_P:
    suggestions.append(f"Phosphorus is low. Use {fertilizer_dict['PLow']}.")
elif P > ideal_P:
    suggestions.append(f"Phosphorus is high. Use {fertilizer_dict['PHigh']} cautiously or reduce its usage.")

if K < ideal_K:
    suggestions.append(f"Potassium is low. Use {fertilizer_dict['KLow']}.")
elif K > ideal_K:
    suggestions.append(f"Potassium is high. Use {fertilizer_dict['KHigh']} cautiously or reduce its usage.")

# Output suggestions
print("\n--- Fertilizer Recommendations ---")
if suggestions:
    for s in suggestions:
        print("-", s)
else:
    print("Your soil NPK levels are optimal for this crop!")
