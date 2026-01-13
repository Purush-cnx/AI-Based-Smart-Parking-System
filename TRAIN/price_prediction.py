import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("D:\FINAL PROJECT\DATA\price data set\Cleaned_Sorted_Agriculture_Data.csv")
#df = df.sample(frac=0.2, random_state=42)  

# Convert date to ordinal
df["Price Date"] = pd.to_datetime(df["Price Date"], errors='coerce')
df["Price_Date_Ordinal"] = df["Price Date"].map(pd.Timestamp.toordinal)

# Encode categorical features
label_encoders = {}
for col in ["STATE", "District Name", "Market Name", "Commodity", "Variety"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df[["STATE", "District Name", "Market Name", "Commodity", "Variety", "Price_Date_Ordinal"]]
y = df["Modal_Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("\nRandom Forest RMSE:", rmse)
print("Model Accuracy (R² Score):", r2 * 100, "%")

# Save model only
joblib.dump(model, r"D:\FINAL PROJECT\MODEL\price_model.pkl")
print("Model saved to random_forest_price_model.pkl")

# Save encoders for later use in prediction
joblib.dump(label_encoders, r"D:\FINAL PROJECT\MODEL\label_encoders.pkl")
print("Encoders saved to label_encoders.pkl")
