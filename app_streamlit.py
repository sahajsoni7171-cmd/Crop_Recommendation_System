import streamlit as st
import numpy as np
import pickle
import os
import pandas as pd

# Load models
model = pickle.load(open("crop_model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
hum_model = pickle.load(open("humidity_model.pkl", "rb"))

# Crop advice dictionary
crop_advice = {
    "rice": ["Needs high rainfall", "Maintain flooded fields", "Warm climate required"],
    "maize": ["Requires well-drained soil", "Moderate rainfall needed", "Needs sunlight"],
    "chickpea": ["Grows in dry climate", "Low water requirement", "Use nitrogen fertilizers"],
    "kidneybeans": ["Needs moderate rainfall", "Well-drained soil", "Avoid waterlogging"],
    "pigeonpeas": ["Drought resistant", "Low water requirement", "Improves soil fertility"],
    "mothbeans": ["Suitable for arid regions", "Needs minimal water", "Grows in sandy soil"],
    "mungbean": ["Requires warm climate", "Moderate rainfall", "Short duration crop"],
    "blackgram": ["Needs well-drained soil", "Low to moderate rainfall", "Avoid excess water"],
    "lentil": ["Cool climate crop", "Requires less water", "Grows in loamy soil"],
    "pomegranate": ["Needs dry climate", "Requires good drainage", "Moderate watering"],
    "banana": ["Requires high humidity", "Needs rich soil", "Frequent watering required"],
    "mango": ["Warm climate required", "Deep soil preferred", "Moderate rainfall"],
    "grapes": ["Needs dry climate", "Requires pruning", "Well-drained soil"],
    "watermelon": ["Needs high temperature", "Sandy soil preferred", "Moderate irrigation"],
    "muskmelon": ["Warm climate crop", "Well-drained soil", "Avoid excess water"],
    "apple": ["Cold climate crop", "Needs chilling period", "Well-drained soil"],
    "orange": ["Requires moderate climate", "Needs irrigation", "Loamy soil preferred"],
    "papaya": ["Warm climate", "Needs sunlight", "Well-drained soil"],
    "coconut": ["High humidity required", "Sandy soil", "Coastal regions preferred"],
    "cotton": ["Requires warm climate", "Moderate rainfall", "Black soil preferred"],
    "jute": ["Needs high rainfall", "Warm and humid climate", "Alluvial soil"],
    "coffee": ["Needs shade", "High humidity", "Well-drained soil"]
}

# UI
st.title("🌾 Crop Recommendation System")
st.write("Enter the Soil and Weather Details Below:")

N = st.number_input("Nitrogen (N)", min_value=0.0)
P = st.number_input("Phosphorus (P)", min_value=0.0)
K = st.number_input("Potassium (K)", min_value=0.0)
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("pH Value")
rainfall = st.number_input("Rainfall (mm)")

if st.button("Predict Crop"):
    data = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    )
    prediction = model.predict(data)
    crop = le.inverse_transform(prediction)[0]
    crop_clean = crop.strip().lower()

    st.success(f"✅ Recommended Crop: {crop_clean}")

    # DEBUG lines - to identify the issue
    st.write("Exact crop name:", repr(crop_clean))
    st.write("Is it in advice dict?", crop_clean in crop_advice)

    if crop_clean in crop_advice:
        st.subheader("🌱 Recommended Practices")
        for tip in crop_advice[crop_clean]:
            st.write("•", tip)
    else:
        st.warning(f"No advice available for '{crop_clean}'")

port = int(os.environ.get("PORT", 8501))
st.write("App running on port", port)
