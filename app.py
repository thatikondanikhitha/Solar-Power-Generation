import streamlit as st
import pickle
import numpy as np
import os

# =========================
# LOAD MODEL & SCALER
# =========================

model = pickle.load(open(os.path.join(os.getcwd(), "solar_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(os.getcwd(), "scaler.pkl"), "rb"))

# =========================
# APP UI
# =========================

st.title("🌞 Solar Power Generation Prediction")

st.write("Enter environmental values to predict solar power")

# =========================
# INPUT FIELDS
# =========================

distance = st.number_input("Distance to Solar Noon", value=0.5)
temperature = st.number_input("Temperature", value=60.0)
wind_direction = st.number_input("Wind Direction", value=25.0)
wind_speed = st.number_input("Wind Speed", value=10.0)
sky_cover = st.number_input("Sky Cover (0–4)", value=2.0)
visibility = st.number_input("Visibility", value=10.0)
humidity = st.number_input("Humidity", value=70.0)
avg_wind = st.number_input("Average Wind Speed", value=10.0)
avg_pressure = st.number_input("Average Pressure", value=30.0)

# =========================
# PREDICT BUTTON
# =========================

if st.button("Predict Solar Power"):

    input_data = np.array([[distance,
                            temperature,
                            wind_direction,
                            wind_speed,
                            sky_cover,
                            visibility,
                            humidity,
                            avg_wind,
                            avg_pressure]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    st.success(f"🔋 Predicted Solar Power: {prediction[0]:.2f}")
