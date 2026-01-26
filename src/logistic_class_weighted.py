import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# -------------------------------------------------
# Paths
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "rfm_with_churn.csv"

# -------------------------------------------------
# Main
# -------------------------------------------------
def train_class_weighted_logistic():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)

    # 2. Features & target (NO Recency)
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

    # 5. Logistic Regression with class weights
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
    model.fit(X_train_scaled, y_train)

    # 6. Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # 7. Evaluation
    print("Classification Report (threshold = 0.5):")
    print(classification_report(y_test, y_pred))

    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))


# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    train_class_weighted_logistic()
