import streamlit as st
import pickle
import numpy as np
import pandas as pd

# CONFIG
st.set_page_config(
    page_title="AI Diabetes Dashboard",
    page_icon="🩺",
    layout="wide"
)

# LOAD MODEL
model = pickle.load(open("saved_model.sav", "rb"))
scaler = pickle.load(open("scaler.sav", "rb"))

# CUSTOM CSS
st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
    background-color:#0f172a;
    color:white;
}

[data-testid="stSidebar"]{
    background-color:#111827;
}

.metric-card {
    background: linear-gradient(135deg,#1e293b,#0f172a);
    padding:20px;
    border-radius:15px;
    text-align:center;
    box-shadow:0 0 10px rgba(0,0,0,0.4);
}

.result-success{
    background:#14532d;
    padding:20px;
    border-radius:12px;
    color:white;
    font-size:24px;
    font-weight:bold;
}

.result-danger{
    background:#7f1d1d;
    padding:20px;
    border-radius:12px;
    color:white;
    font-size:24px;
    font-weight:bold;
}

.stButton>button{
    width:100%;
    height:60px;
    background:linear-gradient(90deg,#2563eb,#7c3aed);
    color:white;
    border:none;
    border-radius:12px;
    font-size:20px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# TITLE
st.title("🩺 AI Diabetes Prediction Dashboard")
st.write("Advanced Machine Learning system for predicting diabetes risk.")

# SIDEBAR
st.sidebar.title("📋 Patient Details")

pregnancies = st.sidebar.slider("Pregnancies",0,20,1)
glucose = st.sidebar.slider("Glucose",0,200,100)
blood_pressure = st.sidebar.slider("Blood Pressure",0,140,70)
skin_thickness = st.sidebar.slider("Skin Thickness",0,100,20)
insulin = st.sidebar.slider("Insulin",0,900,80)
bmi = st.sidebar.slider("BMI",0.0,70.0,25.0)
dpf = st.sidebar.slider("Diabetes Pedigree",0.0,3.0,0.5)
age = st.sidebar.slider("Age",1,100,25)

# METRICS
col1,col2,col3,col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
    <h3>Glucose</h3>
    <h1>{glucose}</h1>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
    <h3>BMI</h3>
    <h1>{bmi}</h1>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
    <h3>Age</h3>
    <h1>{age}</h1>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
    <h3>Insulin</h3>
    <h1>{insulin}</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("## 📊 Health Analytics")

chart_data = pd.DataFrame({
    "Feature":["Glucose","BP","BMI","Insulin"],
    "Value":[glucose,blood_pressure,bmi,insulin]
})

st.bar_chart(chart_data.set_index("Feature"))

# PREDICTION
input_data = np.array([
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    dpf,
    age
]).reshape(1,-1)

scaled_data = scaler.transform(input_data)

if st.button("🚀 Predict Diabetes Risk"):

    prediction = model.predict(scaled_data)

    if prediction[0] == 0:

        st.markdown("""
        <div class="result-success">
        ✅ LOW DIABETES RISK
        </div>
        """, unsafe_allow_html=True)

        st.success("Healthy indicators detected.")

    else:

        st.markdown("""
        <div class="result-danger">
        ⚠️ HIGH DIABETES RISK
        </div>
        """, unsafe_allow_html=True)

        st.error("Consult a healthcare professional.")

# FOOTER
st.markdown("---")
st.caption("Built using Streamlit • Scikit-learn • SVM Classifier • Machine Learning")