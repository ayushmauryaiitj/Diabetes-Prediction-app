import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page Config
st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon="🩺",
    layout="wide"
)

# Load model and scaler
model = pickle.load(open('saved_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }

    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🩺 Diabetes Prediction Dashboard")
st.write("AI-powered Machine Learning application for diabetes risk prediction.")

# Sidebar
st.sidebar.header("Patient Information")

pregnancies = st.sidebar.number_input("Pregnancies", 0.0)
glucose = st.sidebar.number_input("Glucose Level", 0.0)
blood_pressure = st.sidebar.number_input("Blood Pressure", 0.0)
skin_thickness = st.sidebar.number_input("Skin Thickness", 0.0)
insulin = st.sidebar.number_input("Insulin Level", 0.0)
bmi = st.sidebar.number_input("BMI", 0.0)
diabetes_pedigree = st.sidebar.number_input("Diabetes Pedigree Function", 0.0)
age = st.sidebar.number_input("Age", 0.0)

# Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Glucose", glucose)

with col2:
    st.metric("BMI", bmi)

with col3:
    st.metric("Age", age)

# Prediction
if st.button("Predict Diabetes"):

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

    std_data = scaler.transform(input_data)

    prediction = model.predict(std_data)

    st.subheader("Prediction Result")

    if prediction[0] == 0:
        st.success("✅ The person is NOT diabetic")

        st.info("""
        ### Health Suggestions
        - Maintain healthy diet
        - Exercise regularly
        - Monitor glucose levels periodically
        """)

    else:
        st.error("⚠️ The person IS diabetic")

        st.warning("""
        ### Medical Recommendations
        - Consult a healthcare professional
        - Follow proper medication
        - Maintain strict diet control
        """)

# Footer
st.markdown("---")
st.caption("Built with Streamlit, Scikit-learn and Machine Learning")