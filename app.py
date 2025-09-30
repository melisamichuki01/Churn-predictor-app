# full_churn_app.py

import streamlit as st
import pandas as pd
import pickle

# Load model, scaler, and training columns
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📊 Customer Churn Prediction App")

st.write("Provide customer details below to predict churn probability.")

with st.form("churn_form"):

    st.subheader("👤 Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])

    st.subheader("📡 Services")
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.subheader("💰 Account Info")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", 
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1000.0)

    submitted = st.form_submit_button("🔮 Predict")

if submitted:
    # Build dataframe from inputs
    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }])

    # One-hot encode
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    # Scale numeric
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    input_encoded[numeric_cols] = scaler.transform(input_encoded[numeric_cols])
    st.dataframe(input_encoded)

    # Predict
    prediction = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded)[0][1]

    if prediction == 1:
        st.error(f"🚨 Customer likely to CHURN! (Probability: {prob:.2%})")
    else:
        st.success(f"✅ Customer likely to STAY (Churn probability: {prob:.2%})")
