import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
# Load model
try:
    model = pickle.load(open("model.pkl", "rb"))
except:
    st.error("Model file not found. Please upload model.pkl")
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# =========================
# Load model, encoders, features
# =========================
try:
    model = pickle.load(open("model.pkl", "rb"))
    le_dict = pickle.load(open("encoder.pkl", "rb"))
    features = pickle.load(open("features.pkl", "rb"))
except:
    st.error("Model/encoder/features file missing. Please check files.")
    st.stop()

# =========================
# UI
# =========================
st.title("📊 Customer Churn Prediction System")

st.sidebar.title("Customer Churn Predictor")
st.sidebar.write("Upload CSV or use manual prediction")

# =========================
# CSV Upload Section
# =========================
uploaded_file = st.file_uploader("Upload Customer CSV File")

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    # Cleaning
    if "customerID" in data.columns:
        data = data.drop("customerID", axis=1)
    if "Churn" in data.columns:
        data = data.drop("Churn", axis=1)

    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

    # Encoding (correct way)
    for col in data.columns:
        if col in le_dict:
            data[col] = le_dict[col].transform(data[col])

    # Feature alignment
    data = data.reindex(columns=features, fill_value=0)

    # Prediction
    prob = model.predict_proba(data)[:, 1]

    data["Churn_Probability"] = prob

    def risk_level(p):
        if p > 0.7:
            return "High Risk"
        elif p > 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"

    data["Risk_Level"] = data["Churn_Probability"].apply(risk_level)

    # =========================
    # Show Results
    # =========================
    st.subheader("📊 Prediction Results")
    st.dataframe(data)

    # =========================
    # Metrics
    # =========================
    st.subheader("📈 Business Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("High Risk", (data["Risk_Level"]=="High Risk").sum())
    col2.metric("Medium Risk", (data["Risk_Level"]=="Medium Risk").sum())
    col3.metric("Low Risk", (data["Risk_Level"]=="Low Risk").sum())

    # =========================
    # Risk Distribution Chart
    # =========================
    st.subheader("📊 Risk Distribution")
    st.bar_chart(data["Risk_Level"].value_counts())

    # =========================
    # Feature Importance
    # =========================
    st.subheader("📊 Feature Importance")

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots()
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.invert_yaxis()
    st.pyplot(fig)

    # =========================
    # High Risk Customers
    # =========================
    high_risk = data[data["Risk_Level"] == "High Risk"]

    st.subheader("🔴 High Risk Customers")
    st.dataframe(high_risk)

    st.download_button(
        label="Download High Risk Customers",
        data=high_risk.to_csv(index=False),
        file_name="high_risk_customers.csv",
        mime="text/csv"
    )

# =========================
# Manual Prediction Section
# =========================
st.sidebar.subheader("Manual Prediction")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
tenure = st.sidebar.slider("Tenure", 0, 72)
monthly = st.sidebar.number_input("Monthly Charges", 0.0)

if st.sidebar.button("Predict"):

    input_df = pd.DataFrame({
        "gender":[gender],
        "tenure":[tenure],
        "MonthlyCharges":[monthly]
    })

    # Encoding
    for col in input_df.columns:
        if col in le_dict:
            input_df[col] = le_dict[col].transform(input_df[col])

    input_df = input_df.reindex(columns=features, fill_value=0)

    prob = model.predict_proba(input_df)[:,1][0]

    st.sidebar.success(f"Churn Probability: {prob:.2f}")
st.title("📊 Customer Churn Prediction System")

st.sidebar.title("Customer Churn Predictor")
st.sidebar.write("Upload customer dataset to predict churn probability.")

uploaded_file = st.file_uploader("Upload Customer CSV File")



if uploaded_file:

    data = pd.read_csv(uploaded_file)
    
    required_columns = [
    "gender","SeniorCitizen","Partner","Dependents","tenure",
    "PhoneService","MultipleLines","InternetService","OnlineSecurity",
    "OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
    "StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
    "MonthlyCharges","TotalCharges"
]

    missing_cols = [col for col in required_columns if col not in data.columns]

    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        st.stop()

    # Basic cleaning (same as training)
    if "customerID" in data.columns:
        data = data.drop("customerID", axis=1)
    if "Churn" in data.columns:
        data = data.drop("Churn", axis=1)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

    # Encode categorical columns
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = le.fit_transform(data[col])

    # Predict probabilities
    prob = model.predict_proba(data)[:, 1]

    data["Churn_Probability"] = prob

    def risk_level(p):
        if p > 0.7:
            return "High Risk"
        elif p > 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"

    data["Risk_Level"] = data["Churn_Probability"].apply(risk_level)

    st.write("Prediction Results:")
    st.dataframe(data)
    st.subheader("📈 Business Summary")

    total_customers = len(data)
    high_risk_count = len(data[data["Risk_Level"] == "High Risk"])
    medium_risk_count = len(data[data["Risk_Level"] == "Medium Risk"])
    low_risk_count = len(data[data["Risk_Level"] == "Low Risk"])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", total_customers)
    col2.metric("High Risk", high_risk_count)
    col3.metric("Medium Risk", medium_risk_count)
    col4.metric("Low Risk", low_risk_count)
    
    st.subheader("📊 Feature Importance")

    importances = model.feature_importances_
    features = data.drop(["Churn_Probability", "Risk_Level"], axis=1).columns

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots()
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.set_title("Feature Importance")
    ax.invert_yaxis()
    st.pyplot(fig)
        # Show only high risk customers
    high_risk = data[data["Risk_Level"] == "High Risk"]

    st.subheader("🔴 High Risk Customers")
    st.dataframe(high_risk)

    # Download button
    st.download_button(
        label="Download High Risk Customers",
        data=high_risk.to_csv(index=False),
        file_name="high_risk_customers.csv",
        mime="text/csv"
    )