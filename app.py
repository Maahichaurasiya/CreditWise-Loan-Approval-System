import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("models/loan_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Page configuration
st.set_page_config(
    page_title="Smart Loan Approval Hub",
    page_icon="🏦",
    layout="wide"
)

# Custom Dark Professional CSS
st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #111827 50%, #1f2937 100%);
    color: #ffffff;
}

/* Animated background glow */
.stApp::before {
    content: "";
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%);
    animation: rotate 25s linear infinite;
    z-index: -1;
}

@keyframes rotate {
    100% { transform: rotate(360deg); }
}

/* Header */
.header {
    text-align: center;
    padding: 2rem 0;
    animation: fadeInDown 1s ease;
}

.header h1 {
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.header p {
    color: #cbd5e1;
    font-size: 1.1rem;
}

/* Glass card */
.card {
    background: rgba(17, 24, 39, 0.75);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 24px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    margin-bottom: 20px;
    animation: fadeInUp 0.8s ease;
    transition: all 0.3s ease;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(59,130,246,0.15);
}

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, #1e293b, #111827);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 30px rgba(96,165,250,0.2);
}

.metric-card h3 {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-bottom: 8px;
}

.metric-card h2 {
    color: #ffffff;
    font-size: 1.8rem;
    font-weight: 800;
}

/* Button */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    font-size: 17px;
    font-weight: 700;
    border-radius: 14px;
    padding: 12px;
    border: none;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(37,99,235,0.35);
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(124,58,237,0.45);
}

/* Result cards */
.approved {
    background: linear-gradient(145deg, rgba(16,185,129,0.18), rgba(5,150,105,0.18));
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 18px;
    padding: 22px;
    text-align: center;
    animation: pulse 2s infinite;
}

.rejected {
    background: linear-gradient(145deg, rgba(239,68,68,0.18), rgba(185,28,28,0.18));
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 18px;
    padding: 22px;
    text-align: center;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { box-shadow: 0 0 15px rgba(255,255,255,0.05); }
    50% { box-shadow: 0 0 25px rgba(255,255,255,0.12); }
}

/* Animations */
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-30px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Input labels */
label {
    color: #e5e7eb !important;
    font-weight: 600 !important;
}

</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h1>🏦 Smart Loan Approval Hub</h1>
    <p>AI-Powered Loan Approval & Credit Eligibility Prediction</p>
</div>
""", unsafe_allow_html=True)

# Layout
left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Applicant Details")

    applicant_income = st.number_input("Applicant Income", min_value=0.0, value=5000.0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=2000.0)
    age = st.slider("Age", 18, 60, 30)
    dependents = st.slider("Dependents", 0, 5, 1)
    credit_score = st.slider("Credit Score", 300, 900, 700)
    existing_loans = st.slider("Existing Loans", 0, 10, 1)
    dti_ratio = st.slider("DTI Ratio", 0.0, 100.0, 20.0)
    savings = st.number_input("Savings", min_value=0.0, value=50000.0)
    collateral_value = st.number_input("Collateral Value", min_value=0.0, value=100000.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=200000.0)
    loan_term = st.slider("Loan Term (Months)", 6, 360, 120)

    education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    employment = st.selectbox("Employment Status", ["Contract", "Salaried", "Self-employed", "Unemployed"])
    marital = st.selectbox("Marital Status", ["Married", "Single"])
    purpose = st.selectbox("Loan Purpose", ["Business", "Car", "Education", "Home", "Personal"])
    property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    employer = st.selectbox("Employer Category", ["Government", "MNC", "Private", "Unemployed"])

    predict_btn = st.button("🚀 Predict Loan Approval")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Live Dashboard")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Credit Score</h3>
            <h2>{credit_score}</h2>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Loan Amount</h3>
            <h2>₹{loan_amount:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>DTI Ratio</h3>
            <h2>{dti_ratio}%</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#94a3b8; padding:20px;">
    Built with ❤️ using Streamlit and Machine Learning by Maahi Chaurasiya
</div>
""", unsafe_allow_html=True)