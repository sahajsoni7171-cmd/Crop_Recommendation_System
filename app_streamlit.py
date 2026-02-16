import streamlit as st
import numpy as np
import pickle

# Load the trained model and encoder
model = pickle.load(open("crop_model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# Title
st.title("🌾 Crop Recommendation System")

st.write("Enter the Soil and Weather Details Below:")

# Input Fields
N = t1 = st.number_input("Nitrogen (N)", min_value=0.0)
P = t2 = st.number_input("Phosphorus (P)", min_value=0.0)
K = t3 = st.number_input("Potassium (K)", min_value=0.0)
temperature = t4 = st.number_input("Temperature (°C)")
humidity = t5 = st.number_input("Humidity (%)")
ph = t6 = st.number_input("pH Value")
rainfall = t7 = st.number_input("Rainfall (mm)")

# Prediction Button
if st.button("Predict Crop"):
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(data)
    crop = le.inverse_transform(prediction)[0]

    st.success(f"✅ Recommended Crop: {crop}")

    # Needed for Render deployment
import os
port = int(os.environ.get("PORT", 8501))
st.write("App running on port", port)

