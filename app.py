import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load(r'C:\Users\Shree\Downloads\energy\energy\best_energy_model.pkl ')

# Initialize scaler
scaler = StandardScaler()

# Title of the app
st.title("Energy Production Prediction")

# User input fields
temperature = st.number_input("Temperature (°C)", min_value=-30.0, max_value=50.0)
exhaust_vacuum = st.number_input("Exhaust Vacuum (mm Hg)", min_value=0.0, max_value=100.0)
amb_pressure = st.number_input("Ambient Pressure (mbar)", min_value=950.0, max_value=1050.0)
r_humidity = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0)

# Prediction button
if st.button("Predict Energy Production"):
    # Create a DataFrame from user input
    input_data = pd.DataFrame({
        'temperature': [temperature],
        'exhaust_vacuum': [exhaust_vacuum],
        'amb_pressure': [amb_pressure],
        'r_humidity': [r_humidity]
    })

    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)

    # Display the prediction
    st.success(f"Predicted Energy Production: {prediction[0]:.2f} MW")
