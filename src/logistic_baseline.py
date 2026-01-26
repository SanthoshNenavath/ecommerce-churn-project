import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score
)

# -------------------------------------------------
# Paths
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "rfm_with_churn.csv"

# -------------------------------------------------
# Main function
# -------------------------------------------------
def train_and_tune_threshold():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)

    # 2. Features (NO Recency to avoid leakage)
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

    # 4. Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # 6. Predict probabilities
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # 7. ROC AUC (threshold independent)
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC AUC Score: {roc_auc:.4f}\n")

    # 8. Threshold analysis
    print("Threshold analysis (Churn = 1)")
    print("--------------------------------")

    thresholds = [0.5, 0.4, 0.3, 0.25, 0.2]

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        print(
            f"Threshold {threshold:>4} | "
            f"Precision: {precision:.2f} | "
            f"Recall: {recall:.2f}"
        )


# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    train_and_tune_threshold()
