import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
MODEL_PATH = "models/rf_price_model.joblib"
package = joblib.load(MODEL_PATH)
model = package['model']
scaler = package['scaler']
features = package['features']

st.title("📱 Mobile Price Range Prediction App")
st.write("Fill in the mobile specifications below to predict the price category.")

# User Inputs
inputs = {}
for col in features:
    inputs[col] = st.number_input(f"{col}", step=1.0)

# Predict
if st.button("Predict Price Range"):
    data = np.array([list(inputs.values())])
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)[0]

    label = ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"][pred]
    st.success(f"Predicted Price Range: {label}")
