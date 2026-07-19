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

# Header
st.title("🏦 CreditWise Loan Approval Prediction")
st.write("Enter applicant details to predict whether the loan will be approved.")

# Input form
col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input("Applicant Income", min_value=0.0, value=5000.0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=2000.0)
    age = st.slider("Age", 18, 60, 30)
    dependents = st.slider("Dependents", 0, 5, 1)
    credit_score = st.slider("Credit Score", 300, 900, 700)
    existing_loans = st.slider("Existing Loans", 0, 10, 1)

with col2:
    dti_ratio = st.slider("DTI Ratio", 0.0, 100.0, 20.0)
    savings = st.number_input("Savings", min_value=0.0, value=50000.0)
    collateral_value = st.number_input("Collateral Value", min_value=0.0, value=100000.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=200000.0)
    loan_term = st.slider("Loan Term (Months)", 6, 360, 120)

# Categorical inputs
education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
employment = st.selectbox(
    "Employment Status",
    ["Contract", "Salaried", "Self-employed", "Unemployed"]
)
marital = st.selectbox("Marital Status", ["Married", "Single"])
purpose = st.selectbox(
    "Loan Purpose",
    ["Business", "Car", "Education", "Home", "Personal"]
)
property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
gender = st.selectbox("Gender", ["Female", "Male"])
employer = st.selectbox(
    "Employer Category",
    ["Government", "MNC", "Private", "Unemployed"]
)

# Predict button
# Predict button
if st.button("🔍 Predict Loan Approval"):

    # Create input data dictionary
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

        # One-hot encoded features
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

    # Create DataFrame
    input_data = pd.DataFrame([input_dict])

    # Reorder columns to match scaler training data exactly
    input_data = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # Scale input data
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    # Display result
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")

    st.progress(int(probability * 100))
    st.metric("Approval Probability", f"{probability * 100:.2f}%")

    # Show input summary
    st.subheader("Applicant Summary")
    st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)
    # Scale input data
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    # Display result
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")

    st.progress(int(probability * 100))
    st.metric("Approval Probability", f"{probability * 100:.2f}%")

    # Show input summary
    st.subheader("Applicant Summary")
    st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Built with ❤️ using Streamlit and Machine Learning by Maahi Chaurasiya</p>",
    unsafe_allow_html=True
)