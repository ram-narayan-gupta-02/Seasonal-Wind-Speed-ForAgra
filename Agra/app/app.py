# app.py

import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ------------------- Load Model -------------------
model = joblib.load("Agra/model/seasonal_lgbm_model.pkl")

# ------------------- Function to Get Season -------------------
def get_season(month):
    if month in [3, 4, 5, 6]:
        return "Summer"
    elif month in [7, 8, 9, 10]:
        return "Monsoon"
    else:
        return "Winter"

# ------------------- Streamlit UI -------------------
st.title("🌬️ Wind Speed Prediction - Agra (Altitude & Seasonal Model)")
st.markdown("Predict wind speed using historical data for Agra (2020-2025)")

# 📅 Date input
input_date = st.date_input("Select Date", datetime.today())

# ⛰️ Altitude input
altitude = st.number_input("Enter Altitude (in meters)", min_value=0, max_value=26000, value=110)

# 🔘 Predict Button
if st.button("Predict Wind Speed"):
    # Extract date features
    year = input_date.year
    month = input_date.month
    day = input_date.day
    season = get_season(month)

    # Fixed coordinates for Agra
    lat = 27.5
    lon = 77.5

    # One-hot encode season
    season_summer = 1 if season == "Summer" else 0
    season_monsoon = 1 if season == "Monsoon" else 0
    # season_winter = 1 if season == "Winter" else 0

    # Prepare input dataframe
    input_df = pd.DataFrame([{
        "latitude": lat,
        "longitude": lon,
        "year": year,
        "month": month,
        "day": day,
        "altitude(m)": altitude,
        "season_summer": season_summer,
        "season_monsoon": season_monsoon
    }])

    # Predict wind speed
    prediction = model.predict(input_df)[0]

    # Show results
    st.success(f"🌡️ Predicted Wind Speed: **{prediction:.2f} m/s**")
    st.info(f"🚗 Predicted Wind Speed: **{prediction * 3.6:.2f} km/h**")
