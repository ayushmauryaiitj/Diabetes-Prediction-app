import streamlit as st
import pickle
import numpy as np

# Load model and scaler
loaded_model = pickle.load(open('saved_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# Title
st.title("Diabetes Prediction Web App")

st.write("Enter the patient's medical details below:")

# Input fields
pregnancies = st.number_input("Pregnancies")
glucose = st.number_input("Glucose Level")
blood_pressure = st.number_input("Blood Pressure")
skin_thickness = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin Level")
bmi = st.number_input("BMI")
diabetes_pedigree = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

# Prediction button
if st.button("Predict"):

    input_data = np.array([
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]).reshape(1, -1)

    # Standardize
    std_data = scaler.transform(input_data)

    # Prediction
    prediction = loaded_model.predict(std_data)

    if prediction[0] == 0:
        st.success("The person is NOT diabetic")
    else:
        st.error("The person is diabetic")