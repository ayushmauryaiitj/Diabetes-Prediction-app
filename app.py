import streamlit as st
import pickle
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Diabetes AI Dashboard",
    page_icon="🩺",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("saved_model.sav", "rb"))
scaler = pickle.load(open("scaler.sav", "rb"))

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}

.block-container {
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

.metric-card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    text-align: center;
    transition: 0.3s;
}

.metric-card:hover {
    transform: translateY(-3px);
}

.metric-title {
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 42px;
    font-weight: 700;
    color: #111827;
}

.stButton > button {
    width: 100%;
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 12px;
    height: 55px;
    font-size: 18px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #1d4ed8;
    color: white;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    font-size: 24px;
    font-weight: bold;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🩺 Patient Information")

pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 1)
glucose = st.sidebar.slider("Glucose Level", 0, 200, 100)
blood_pressure = st.sidebar.slider("Blood Pressure", 0, 140, 70)
skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin Level", 0, 900, 80)
bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree", 0.0, 3.0, 0.5)
age = st.sidebar.slider("Age", 1, 100, 25)

# =========================
# HEADER
# =========================
st.title("🩺 Diabetes Prediction Dashboard")
st.write("AI-powered Machine Learning application for diabetes risk analysis.")

# =========================
# METRIC CARDS
# =========================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Glucose</div>
        <div class="metric-value">{glucose}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">BMI</div>
        <div class="metric-value">{bmi}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Age</div>
        <div class="metric-value">{age}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Insulin</div>
        <div class="metric-value">{insulin}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# =========================
# MAIN SECTION
# =========================
left, right = st.columns([1.3, 1])

# =========================
# HEALTH SUMMARY
# =========================
with left:

    st.subheader("📊 Health Summary")

    st.write("### Glucose Level")
    st.progress(min(glucose / 200, 1.0))
    st.caption(f"{glucose} / 200")

    st.write("### BMI")
    st.progress(min(bmi / 70, 1.0))
    st.caption(f"{bmi} / 70")

    st.write("### Insulin Level")
    st.progress(min(insulin / 900, 1.0))
    st.caption(f"{insulin} / 900")

    st.write("### Age")
    st.progress(min(age / 100, 1.0))
    st.caption(f"{age} / 100")

# =========================
# PREDICTION PANEL
# =========================
with right:

    st.subheader("🧠 Prediction Panel")

    input_data = np.array([
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]).reshape(1, -1)

    scaled_data = scaler.transform(input_data)

    if st.button("Predict Diabetes Risk"):

        prediction = model.predict(scaled_data)

        if prediction[0] == 0:

            st.markdown("""
            <div class="result-box"
            style="background:#dcfce7;color:#166534;">
            ✅ LOW RISK
            </div>
            """, unsafe_allow_html=True)

            st.success("Patient indicators appear healthy.")

        else:

            st.markdown("""
            <div class="result-box"
            style="background:#fee2e2;color:#991b1b;">
            ⚠️ HIGH RISK
            </div>
            """, unsafe_allow_html=True)

            st.error("Patient may have diabetes risk.")

# =========================
# FOOTER
# =========================
st.write("")
st.markdown("---")
st.caption(
    "Built with Streamlit • Scikit-learn • SVM Classifier • Machine Learning"
)