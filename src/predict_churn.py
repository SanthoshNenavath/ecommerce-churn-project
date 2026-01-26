import joblib
from pathlib import Path
import numpy as np

# -------------------------------------------------
# Paths
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_logistic_model.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"

# -------------------------------------------------
# Load model & scaler
# -------------------------------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -------------------------------------------------
# Prediction function
# -------------------------------------------------
def predict_churn(frequency: int, monetary: float, threshold: float = 0.5):
    """
    Predict churn probability and decision for a customer.
    """
    # Create input array
    X = np.array([[frequency, monetary]])

    # Scale input
    X_scaled = scaler.transform(X)

    # Predict probability
    churn_proba = model.predict_proba(X_scaled)[0][1]

    # Decision
    churn_label = int(churn_proba >= threshold)

    return churn_proba, churn_label


# -------------------------------------------------
# Example run
# -------------------------------------------------
if __name__ == "__main__":
    # Example customer
    frequency = 2
    monetary = 500.0

    prob, label = predict_churn(frequency, monetary)

    print("Customer Input:")
    print(f"Frequency: {frequency}, Monetary: {monetary}")

    print("\nPrediction:")
    print(f"Churn Probability: {prob:.2f}")
    print("Churn Decision:", "YES" if label == 1 else "NO")
