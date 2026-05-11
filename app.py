import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load("housing_price_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("California House Price Predictor")

col1, col2 = st.columns(2)
with col1:
    longitude = st.number_input("Longitude", value=-118.25)
    latitude = st.number_input("Latitude", value=34.05)
    housing_median_age = st.slider("Housing Median Age", 1, 52, 28)
    total_rooms = st.number_input("Total Rooms", value=2635)

with col2:
    total_bedrooms = st.number_input("Total Bedrooms", value=537)
    population = st.number_input("Population", value=1425)
    households = st.number_input("Households", value=499)
    median_income = st.number_input("Median Income (in $10k)", value=3.87)

ocean = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])

if st.button("Predict Price"):
    input_dict = {
        "longitude": longitude, "latitude": latitude,
        "housing_median_age": housing_median_age, "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms, "population": population,
        "households": households, "median_income": median_income,
        "<1H OCEAN": 0, "INLAND": 0, "ISLAND": 0, "NEAR BAY": 0, "NEAR OCEAN": 0
    }
    input_dict[ocean] = 1
    df = pd.DataFrame([input_dict]).reindex(columns=feature_columns, fill_value=0)
    pred = model.predict(df)[0]
    st.success(f"Estimated House Value: **${pred:,.0f}**")