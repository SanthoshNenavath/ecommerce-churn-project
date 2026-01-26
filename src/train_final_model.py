import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# -------------------------------------------------
# Paths
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "rfm_with_churn.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "churn_logistic_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# -------------------------------------------------
# Train & Save Final Model
# -------------------------------------------------
def train_and_save_final_model():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)

    # 2. Features & target (NO Recency to avoid leakage)
    X = df[["Frequency", "Monetary"]]
    y = df["Churn"]

    # 3. Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 4. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Final Logistic Regression model
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
    model.fit(X_train_scaled, y_train)

    # 6. Evaluation (for record)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("Final Model Evaluation:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

    # 7. Save model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("\nModel saved to:", MODEL_PATH)
    print("Scaler saved to:", SCALER_PATH)


# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    train_and_save_final_model()
