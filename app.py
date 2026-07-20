import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie

# 1. Page Configuration
st.set_page_config(
    page_title="CreditWise Risk Portal",
    page_icon="⚡",
    layout="wide"
)

# 2. Advanced Premium Glassmorphism CSS Styling
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Dynamic Typography */
    h1 {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    /* Neon Action Button */
    .stButton>button {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%) !important;
        color: #0f172a !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0px 4px 20px rgba(0, 242, 254, 0.3);
        transition: all 0.3s ease-in-out !important;
        width: 100%;
        padding: 14px 0px;
        font-size: 16px !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 6px 25px rgba(0, 242, 254, 0.5);
    }
    
    /* Metrics Customization */
    div[data-testid="stMetricValue"] {
        font-size: 28px !important;
        color: #00f2fe !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Handle Lottie 3D Animation Asset
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# Premium 3D finance/credit layout asset
lottie_url = "https://assets5.lottiefiles.com/packages/lf20_q5pk6p1k.json" 
lottie_json = load_lottieurl(lottie_url)

# 4. Model & Scaler Loading (Using your paths)
@st.cache_resource
def load_model():
    try:
        with open("models/loan_model.pkl", "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None

@st.cache_resource
def load_scaler():
    try:
        with open("models/scaler.pkl", "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None

model = load_model()
scaler = load_scaler()

# 5. Dashboard Hero Header
col_header, col_hero_anim = st.columns([2, 1])
with col_header:
    st.title("⚡ CREDITWISE LOAN ANALYTICS PLATFORM")
    st.markdown("##### Enterprise Risk Architecture & Predictive Underwriting Machine Learning Dashboard")
    st.caption("Adjust target metrics inside the data matrices to verify applicant threshold compatibility.")
with col_hero_anim:
    if lottie_json:
        st_lottie(lottie_json, height=150, key="hero_anim")

st.write("---")

# 6. Structured User Inputs (Organized across 3 neat workspace panels)
st.markdown("### 📊 Underwriting Data Matrix")
panel1, panel2, panel3 = st.columns(3)

with panel1:
    st.markdown("**💵 Financial Assets**")
    applicant_income = st.number_input("Applicant Income ($)", min_value=0.0, value=20000.0, step=1000.0)
    coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0.0, value=10000.0, step=1000.0)
    savings = st.number_input("Total Savings Balance ($)", min_value=0.0, value=500000.0, step=5000.0)
    collateral_value = st.number_input("Collateral Fair Market Value ($)", min_value=0.0, value=1000000.0, step=10000.0)

with panel2:
    st.markdown("**🛡️ Risk Profile Indicators**")
    credit_score = st.slider("Credit Bureau Score", 300, 900, 900)
    dti_ratio = st.slider("DTI (Debt-to-Income) Ratio %", 0.0, 100.0, 5.0, step=0.5)
    existing_loans = st.slider("Active Existing Loans Count", 0, 10, 0)
    age = st.slider("Applicant Age", 18, 60, 30)
    dependents = st.slider("Dependents Count", 0, 5, 0)

with panel3:
    st.markdown("**📝 Demographics & Terms**")
    loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0.0, value=50000.0, step=5000.0)
    loan_term = st.slider("Loan Amortization Term (Months)", 6, 360, 12)
    
    # Categorical Select Boxes directly derived from your encoding structure
    gender = st.selectbox("Gender Orientation", ["Male", "Female"])
    marital = st.selectbox("Marital Status", ["Married", "Single"])
    education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    employment = st.selectbox("Employment Structure", ["Salaried", "Self-employed", "Contract", "Unemployed"])
    employer = st.selectbox("Employer Ecosystem", ["Private", "MNC", "Government", "Unemployed"])
    purpose = st.selectbox("Loan Purpose Categorization", ["Home", "Business", "Personal", "Education", "Car"])
    property_area = st.selectbox("Zoning Location Matrix", ["Urban", "Semiurban", "Rural"])

st.write("---")

# 7. Execution Engine & Dashboard Generation Block
if st.button("🚀 EXECUTE FINANCIAL MODEL RISK ASSESSMENT"):
    # Reconstructing the exact mapping dictionary structural parameters from your script
    input_dict = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Education_Level": 1 if education == "Not Graduate" else 0,
        "Employment_Status_Salaried": 1 if employment == "Salaried" else 0,
        "Employment_Status_Self-employed": 1 if employment == "Self-employed" else 0,
        "Employment_Status_Unemployed": 1 if employment == "Unemployed" else 0,
        "Marital_Status_Single": 1 if marital == "Single" else 0,
        "Loan_Purpose_Car": 1 if purpose == "Car" else 0,
        "Loan_Purpose_Education": 1 if purpose == "Education" else 0,
        "Loan_Purpose_Home": 1 if purpose == "Home" else 0,
        "Loan_Purpose_Personal": 1 if purpose == "Personal" else 0,
        "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
        "Property_Area_Urban": 1 if property_area == "Urban" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Employer_Category_Government": 1 if employer == "Government" else 0,
        "Employer_Category_MNC": 1 if employer == "MNC" else 0,
        "Employer_Category_Private": 1 if employer == "Private" else 0,
        "Employer_Category_Unemployed": 1 if employer == "Unemployed" else 0
    }

    try:
        input_data = pd.DataFrame([input_dict])

        if model is not None and scaler is not None:
            # Reindex features to map scaler requirements exactly
            input_data = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)
            
            # Scaler Transformation matrix matching step
            scaled_input = scaler.transform(input_data)
            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0][1]
        else:
            # UI Safe Sandbox Mock Mode if paths are not locally found
            prediction = 1 if (credit_score >= 650 and dti_ratio < 45.0) else 0
            probability = 0.88 if prediction == 1 else 0.15
            st.warning("⚠️ Running inside UI Standalone Preview Engine Sandbox (Files missing in models/)")

        # --- THE ADVANCED VISUAL DASHBOARD ---
        st.markdown("### 📈 Real-Time Assessment Dashboard Insights")
        
        # Row 1: KPI Analytics Panel Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        total_income = applicant_income + coapplicant_income
        leverage_ratio = loan_amount / max(total_income, 1.0)
        
        with kpi1:
            st.metric("Risk Status Matrix", "APPROVED" if prediction == 1 else "DECLINED", 
                      delta="Compliance Target Cleared" if prediction == 1 else "High Structural Risk")
        with kpi2:
            st.metric("Model Confidence Rating", f"{probability*100:.2f}%")
        with kpi3:
            st.metric("Consolidated Liquidity", f"${total_income:,.2f}")
        with kpi4:
            st.metric("Loan Asset Leverage Ratio", f"{leverage_ratio:.2f}x")

        st.write("")

        # Row 2: Advanced Interactive Visualization Charts (Plotly Component Grid)
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Gauge Vector Meter Chart representing exact probabilities
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Underwriting Security Margin Rating (%)", 'font': {'color': '#f8fafc', 'size': 16}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#f8fafc"},
                    'bar': {'color': "#00f2fe"},
                    'steps': [
                        {'range': [0, 45], 'color': 'rgba(239, 68, 68, 0.25)'},
                        {'range': [45, 75], 'color': 'rgba(245, 158, 11, 0.25)'},
                        {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.25)'}
                    ],
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#f8fafc"}, height=300, margin=dict(t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with chart_col2:
            # Asset Allocation and Exposure Chart Comparing Liquidity Vs Requested Value
            chart_data = pd.DataFrame({
                'Capital Segments': ['Base Income Pool', 'Coapplicant Contribution', 'Savings Liquidity', 'Requested Funding'],
                'Value ($)': [applicant_income, coapplicant_income, savings, loan_amount]
            })
            fig_bar = px.bar(
                chart_data, x='Capital Segments', y='Value ($)', 
                title="Financial Liquidity Vs Asset Request Ratios",
                color='Capital Segments',
                color_discrete_sequence=['#4facfe', '#00f2fe', '#10b981', '#f43f5e']
            )
            fig_bar.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#f8fafc"}, height=300, margin=dict(t=40, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

    except Exception as e:
        st.error(f"Engine Exception Encountered: {e}")