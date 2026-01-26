import pandas as pd
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[1]

# Input and output paths
RFM_PATH = ROOT / "data" / "processed" / "rfm.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "rfm_with_churn.csv"

def add_churn_label(rfm: pd.DataFrame, threshold: int = 180) -> pd.DataFrame:
    """
    Adds a binary churn label.
    Churn = 1 if Recency > threshold, else 0
    """
    rfm = rfm.copy()
    rfm["Churn"] = (rfm["Recency"] > threshold).astype(int)
    return rfm

if __name__ == "__main__":
    # Load RFM data
    rfm = pd.read_csv(RFM_PATH)

    print("Original RFM shape:", rfm.shape)

    # Add churn label
    rfm_churn = add_churn_label(rfm, threshold=180)

    print("\nChurn distribution:")
    print(rfm_churn["Churn"].value_counts())

    print("\nChurn percentage:")
    print(rfm_churn["Churn"].value_counts(normalize=True) * 100)

    # Save output
    rfm_churn.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved file to: {OUTPUT_PATH}")
