import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================
# 1. Load Data
# =========================
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop unnecessary column
df.drop("customerID", axis=1, inplace=True)

# Fix TotalCharges column
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# =========================
# 2. Encode Categorical Columns (FIXED)
# =========================
le_dict = {}

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

# Save encoders
pickle.dump(le_dict, open("encoder.pkl", "wb"))



# =========================
# 3. Split Features & Target
# =========================
X = df.drop("Churn", axis=1)
y = df["Churn"]

# =========================
# 4. Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. Train Model
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# =========================
# 6. Evaluation
# =========================


y_prob = model.predict_proba(X_test)[:, 1]

threshold = 0.35
y_pred = (y_prob > threshold).astype(int)

results = X_test.copy()
results["Actual_Churn"] = y_test.values
results["Predicted_Churn"] = y_pred
results["Churn_Probability"] = y_prob

# Risk categorization
def risk_level(prob):
    if prob > 0.7:
        return "High Risk"
    elif prob > 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

results["Risk_Level"] = results["Churn_Probability"].apply(risk_level)

print(results.head())
#Save Feature Column
pickle.dump(X.columns.tolist(), open("features.pkl", "wb"))
# =========================
# 7. Save Model
# =========================
pickle.dump(model, open("model.pkl", "wb"))

print("\nModel saved as model.pkl")