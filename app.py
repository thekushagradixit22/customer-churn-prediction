import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("📊 Customer Churn Prediction System")

uploaded_file = st.file_uploader("Upload Customer CSV File")

if uploaded_file:

    data = pd.read_csv(uploaded_file)

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

    st.bar_chart(importance_df.set_index("Feature"))
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