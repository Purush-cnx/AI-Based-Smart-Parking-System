import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load model and encoders
model = joblib.load("D:\FINAL PROJECT\MODEL\price_model.pkl")
label_encoders = joblib.load("D:/FINAL PROJECT/MODEL/label_encoders.pkl")

# Load dataset for mapping
df = pd.read_csv("D:/FINAL PROJECT/DATA/price data set/Cleaned_Sorted_Agriculture_Data.csv")

# ==== Farmer Inputs ====
input_state = input("\nEnter State: ").strip()
input_district = input("Enter District: ").strip()
input_commodity = input("Enter Commodity: ").strip()
input_weight = input("Enter Available Quantity (in quintals) [optional]: ").strip()
input_cost = input("Enter Cost of Growing (₹ per quintal) [optional]: ").strip()

# Convert to appropriate types
input_weight = float(input_weight) if input_weight else 1.0
input_cost = float(input_cost) if input_cost else 0.0

# ==== Encode inputs ====
try:
    state_encoded = label_encoders["STATE"].transform([input_state])[0]
    district_encoded = label_encoders["District Name"].transform([input_district])[0]
    commodity_encoded = label_encoders["Commodity"].transform([input_commodity])[0]
except Exception as e:
    print(f"\nInput Error: {e}")
    exit()

# Prepare encoded columns in data
df["STATE"] = label_encoders["STATE"].transform(df["STATE"])
df["District Name"] = label_encoders["District Name"].transform(df["District Name"])
df["Commodity"] = label_encoders["Commodity"].transform(df["Commodity"])
df["Variety"] = label_encoders["Variety"].transform(df["Variety"])
df["Market Name"] = label_encoders["Market Name"].transform(df["Market Name"])

# ==== Get Most Common Variety and Market for Entered Commodity & District ====
filtered = df[(df["District Name"] == district_encoded) & (df["Commodity"] == commodity_encoded)]
if filtered.empty:
    print("No data found for this combination.")
    exit()

variety_encoded = filtered["Variety"].mode()[0]
market_encoded = filtered["Market Name"].mode()[0]

# Today's date as ordinal
today_ordinal = datetime.today().toordinal()

# ==== Predict Price in Entered District ====
input_features = pd.DataFrame([[
    state_encoded,
    district_encoded,
    market_encoded,
    commodity_encoded,
    variety_encoded,
    today_ordinal
]], columns=[
    "STATE",
    "District Name",
    "Market Name",
    "Commodity",
    "Variety",
    "Price_Date_Ordinal"
])

predicted_price = model.predict(input_features)[0]
estimated_income = predicted_price * input_weight
net_profit = (predicted_price - input_cost) * input_weight

print("\nYour Local Market:")
print(f"Predicted Modal Price for {input_commodity} in {input_district}: ₹{round(predicted_price, 2)} per quintal")
print(f"Total Estimated Value for {input_weight} quintals: ₹{round(estimated_income, 2)}")
if input_cost > 0:
    print(f"Estimated Net Profit after growing cost (₹{input_cost}/qtl): ₹{round(net_profit, 2)}")

# ==== Find Other Best District + Market Options in the Same State ====
unique_combos = df[df["STATE"] == state_encoded][["District Name", "Market Name"]].drop_duplicates()

results = []
for _, row in unique_combos.iterrows():
    dist, market = row["District Name"], row["Market Name"]

    # Get most common variety for this district + commodity
    sub_df = df[(df["District Name"] == dist) & (df["Commodity"] == commodity_encoded)]
    if sub_df.empty:
        continue
    variety = sub_df["Variety"].mode()[0]

    # Predict price
    row_input = pd.DataFrame([[
        state_encoded,
        dist,
        market,
        commodity_encoded,
        variety,
        today_ordinal
    ]], columns=[
        "STATE",
        "District Name",
        "Market Name",
        "Commodity",
        "Variety",
        "Price_Date_Ordinal"
    ])

    pred_price = model.predict(row_input)[0]
    dist_name = label_encoders["District Name"].inverse_transform([dist])[0]
    market_name = label_encoders["Market Name"].inverse_transform([market])[0]
    extra_profit = pred_price - predicted_price
    results.append((dist_name, market_name, pred_price, extra_profit))

# ==== Sort and Display Top 3 ====
top_3 = sorted(results, key=lambda x: x[2], reverse=True)[:3]

print(f"\nTop 3 Market Recommendations in {input_state}")
for i, (dist, market, price, gain) in enumerate(top_3, 1):
    total_value = price * input_weight
    print(f"{i}. {market}, {dist}: ₹{round(price, 2)} per quintal → ₹{round(total_value, 2)} total (↑ ₹{round(gain, 2)}/qtl)")

best = top_3[0]
print(f"\nBest Option: {best[1]}, {best[0]}")
print(f"Max Estimated Income: ₹{round(best[2] * input_weight, 2)} for {input_weight} quintals")
