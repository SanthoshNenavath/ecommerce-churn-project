from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# -------------------------------------------------
# App initialization
# -------------------------------------------------
app = FastAPI(
    title="E-Commerce Churn Prediction API",
    description="Predicts customer churn using a trained ML model",
    version="1.0"
)

# -------------------------------------------------
# Load & train model ON STARTUP
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "rfm_with_churn.csv"

# Load data
df = pd.read_csv(DATA_PATH)

X = df[["Frequency", "Monetary"]]
y = df["Churn"]

# Train-test split
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
model.fit(X_train_scaled, y_train)

# -------------------------------------------------
# Request schema
# -------------------------------------------------
class CustomerInput(BaseModel):
    frequency: int
    monetary: float

# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------
@app.post("/predict")
def predict_churn(data: CustomerInput):
    X = np.array([[data.frequency, data.monetary]])
    X_scaled = scaler.transform(X)

    churn_proba = model.predict_proba(X_scaled)[0][1]
    threshold = 0.5

    return {
        "churn_probability": round(float(churn_proba), 4),
        "churn_prediction": "YES" if churn_proba >= threshold else "NO",
        "threshold_used": threshold
    }

# -------------------------------------------------
# Health check
# -------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "API is running"}
