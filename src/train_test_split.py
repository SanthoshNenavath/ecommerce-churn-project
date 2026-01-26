import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project root
ROOT = Path(__file__).resolve().parents[1]

# Input file
DATA_PATH = ROOT / "data" / "processed" / "rfm_with_churn.csv"

def split_data(test_size=0.2, random_state=42):
    # Load data
    df = pd.read_csv(DATA_PATH)

    # Features and target
    X = df[["Recency", "Frequency", "Monetary"]]
    y = df["Churn"]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = split_data()

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    print("\nTrain churn distribution:")
    print(y_train.value_counts(normalize=True) * 100)

    print("\nTest churn distribution:")
    print(y_test.value_counts(normalize=True) * 100)
