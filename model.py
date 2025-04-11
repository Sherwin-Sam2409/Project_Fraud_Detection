import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import ipaddress

# Constants
MODEL_FILE = "fraud_xgboost.pkl"
ENCODER_FILE = "label_encoders.pkl"
SCALER_FILE = "scaler.pkl"
FEATURE_ORDER_FILE = "feature_order.pkl"
THRESHOLD = 0.1

def is_private_ip(ip):
    try:
        return ipaddress.ip_address(ip).is_private
    except:
        return True

def extract_ip_features(df):
    df["Is_Private_IP"] = df["IP Address"].apply(is_private_ip).astype(int)
    ip_counts = df["IP Address"].value_counts().to_dict()
    df["IP_Transaction_Count"] = df["IP Address"].map(ip_counts)
    df["New_IP"] = df.groupby("Sender Account ID")["IP Address"].transform(lambda x: x.duplicated().astype(int))
    df.drop(columns=["IP Address"], inplace=True)
    return df

def preprocess_data(df, training=True):
    if df.empty:
        return None, None

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Hour"] = df["Timestamp"].dt.hour
        df["Day"] = df["Timestamp"].dt.dayofweek
        df.drop(columns=["Timestamp"], inplace=True)
    else:
        df["Hour"] = 0
        df["Day"] = 0

    df.drop(columns=["Transaction ID"], errors="ignore", inplace=True)
    df = extract_ip_features(df)

    categorical = [
        "Sender Account ID", "Receiver Account ID", "Type", "Device ID",
        "Location", "User Device Recognition", "Known Threat Flag",
        "Login Location Match", "Spending Pattern Deviation"
    ]

    if training:
        encoders = {col: LabelEncoder().fit(df[col]) for col in categorical}
        joblib.dump(encoders, ENCODER_FILE)
    else:
        encoders = joblib.load(ENCODER_FILE)

    for col in categorical:
        if col in df.columns:
            if training:
                df[col] = encoders[col].transform(df[col])
            else:
                df[col] = df[col].map(lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else 0)

    numeric = ["Amount", "Account Balance", "Hour", "Day", "IP_Transaction_Count", "New_IP"]

    if training:
        scaler = MinMaxScaler()
        df[numeric] = scaler.fit_transform(df[numeric])
        joblib.dump(scaler, SCALER_FILE)
    else:
        scaler = joblib.load(SCALER_FILE)
        df[numeric] = scaler.transform(df[numeric])

    if training:
        features = df.drop(columns=["Suspicious Activity Flag"], errors="ignore")
        joblib.dump(list(features.columns), FEATURE_ORDER_FILE)
        return features, df["Suspicious Activity Flag"]
    else:
        feature_order = joblib.load(FEATURE_ORDER_FILE)
        df = df.reindex(columns=feature_order, fill_value=0)
        return df

def plot_accuracy_histogram(y_true, y_pred):
    plt.figure(figsize=(8, 5))
    sns.histplot(y_true == y_pred, bins=2, color="blue")
    plt.xlabel("Prediction Correctness")
    plt.ylabel("Count")
    plt.title("Prediction Accuracy")
    plt.xticks([0, 1], ["Incorrect", "Correct"])
    plt.show()

def train_model(csv_file):
    df = pd.read_csv(csv_file)
    df.dropna(subset=["Suspicious Activity Flag"], inplace=True)
    df["Suspicious Activity Flag"] = df["Suspicious Activity Flag"].astype(int)

    X, y = preprocess_data(df, training=True)

    # Show fraud/legit count
    print(f"Fraud cases: {sum(y)}, Legit cases: {len(y) - sum(y)}")

    # Handle imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    model = XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILE)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model trained. Accuracy: {acc * 100:.2f}%")
    plot_accuracy_histogram(y_test, y_pred)

def detect_fraud(transaction_data):
    if not os.path.exists(MODEL_FILE):
        return {"error": "Model not found. Train it first."}
    try:
        model = joblib.load(MODEL_FILE)
        df = pd.DataFrame([transaction_data])
        X = preprocess_data(df, training=False)
        prob = model.predict_proba(X)[0][1]
        fraudulent = prob >= THRESHOLD

        return {
            "fraudulent": bool(fraudulent),
            "confidence": float(np.round(prob, 6)),
            "message": "Fraudulent transaction detected." if fraudulent else "Transaction appears legitimate."
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

if __name__ == "__main__":
    train_model("financial_transactions.csv")
