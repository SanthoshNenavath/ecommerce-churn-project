from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

# -------------------------------------------------
# App initialization
# -------------------------------------------------
app = FastAPI(
    title="E-Commerce Churn Prediction API",
    description="Predicts customer churn using a trained ML model",
    version="1.0"
)

# -------------------------------------------------
# Paths
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_logistic_model.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"

# -------------------------------------------------
# Load model and scaler
# -------------------------------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

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
    """
    Predict churn probability and decision.
    """
    # Prepare input
    X = np.array([[data.frequency, data.monetary]])

    # Scale
    X_scaled = scaler.transform(X)

    # Predict probability
    churn_proba = model.predict_proba(X_scaled)[0][1]

    # Decision threshold (locked)
    threshold = 0.5
    churn_label = int(churn_proba >= threshold)

    return {
        "churn_probability": round(float(churn_proba), 4),
        "churn_prediction": "YES" if churn_label == 1 else "NO",
        "threshold_used": threshold
    }

# -------------------------------------------------
# Health check
# -------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "API is running"}
