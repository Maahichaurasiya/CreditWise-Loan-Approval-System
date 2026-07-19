import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load("models/loan_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Page configuration
st.set_page_config(
    page_title="CreditWise Loan Approval",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS for modern UI
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

.header {
    text-align: center;
    padding: 2rem 0;
}

.header h1 {
    font-size: 3rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.5rem;
}

.header p {
    font-size: 1.1rem;
    color: #cbd5e1;
}

.glass-card {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 25px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    margin-bottom: 20px;
}

.metric-card {
    background: rgba(255, 255, 255, 0.12);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.15);
}

.metric-card h3 {
    color: #cbd5e1;
    font-size: 1rem;
    margin-bottom: 10px;
}

.metric-card h2 {
    color: #ffffff;
    font-size: 2rem;
    font-weight: 700;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #3b82f6, #2563eb);
    color: white;
    font-size: 18px;
    font-weight: 700;
    border-radius: 12px;
    padding: 12px;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
}

.result-card {
    text-align: center;
    padding: 25px;
    border-radius: 18px;
    margin-top: 20px;
}

.approved {
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.4);
}

.rejected {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.4);
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h1>🏦 CreditWise Loan Approval</h1>
    <p>Smart AI-powered loan approval prediction system</p>
</div>
""", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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

    predict_btn = st.button("🔍 Predict Loan Approval")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📊 Quick Overview")

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

    if predict_btn:
        input_data = pd.DataFrame({
            "Applicant_Income": [applicant_income],
            "Coapplicant_Income": [coapplicant_income],
            "Age": [age],
            "Dependents": [dependents],
            "Credit_Score": [credit_score],
            "Existing_Loans": [existing_loans],
            "DTI_Ratio": [dti_ratio],
            "Savings": [savings],
            "Collateral_Value": [collateral_value],
            "Loan_Amount": [loan_amount],
            "Loan_Term": [loan_term],
            "Education_Level": [1 if education == "Not Graduate" else 0],

            "Employment_Status_Salaried": [1 if employment == "Salaried" else 0],
            "Employment_Status_Self-employed": [1 if employment == "Self-employed" else 0],
            "Employment_Status_Unemployed": [1 if employment == "Unemployed" else 0],
            "Marital_Status_Single": [1 if marital == "Single" else 0],
            "Loan_Purpose_Car": [1 if purpose == "Car" else 0],
            "Loan_Purpose_Education": [1 if purpose == "Education" else 0],
            "Loan_Purpose_Home": [1 if purpose == "Home" else 0],
            "Loan_Purpose_Personal": [1 if purpose == "Personal" else 0],
            "Property_Area_Semiurban": [1 if property_area == "Semiurban" else 0],
            "Property_Area_Urban": [1 if property_area == "Urban" else 0],
            "Gender_Male": [1 if gender == "Male" else 0],
            "Employer_Category_Government": [1 if employer == "Government" else 0],
            "Employer_Category_MNC": [1 if employer == "MNC" else 0],
            "Employer_Category_Private": [1 if employer == "Private" else 0],
            "Employer_Category_Unemployed": [1 if employer == "Unemployed" else 0]
        })

        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]

        if prediction == 1:
            st.markdown(f"""
            <div class="result-card approved">
                <h2>✅ Loan Approved</h2>
                <h3>Approval Probability: {probability*100:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card rejected">
                <h2>❌ Loan Rejected</h2>
                <h3>Approval Probability: {probability*100:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)

        st.progress(int(probability * 100))
        st.subheader("📋 Applicant Summary")
        st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#cbd5e1; padding:20px;">
    Built with ❤️ using Streamlit and Machine Learning by Maahi Chaurasiya
</div>
""", unsafe_allow_html=True)