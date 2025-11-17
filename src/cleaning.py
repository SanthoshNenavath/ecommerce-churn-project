import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "online_retail.csv"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


def load_raw():
    print(f"Loading raw dataset from: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH, encoding="ISO-8859-1")
    print("Raw shape =", df.shape)
    return df


def clean_data(df):
    print("\nDropping missing CustomerID...")
    df = df.dropna(subset=["CustomerID"])

    print("Removing cancelled orders (InvoiceNo starting with 'C')...")
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    print("Converting InvoiceDate to datetime...")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    print("Dropping rows with invalid dates...")
    df = df.dropna(subset=["InvoiceDate"])

    print("Creating TotalPrice = Quantity * UnitPrice...")
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    print("Final cleaned shape =", df.shape)
    return df


def create_rfm(df):
    print("\nCreating RFM table...")

    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (reference_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum"
    })

    rfm.rename(columns={
        "InvoiceDate": "Recency",
        "InvoiceNo": "Frequency",
        "TotalPrice": "Monetary"
    }, inplace=True)

    print("RFM shape =", rfm.shape)
    return rfm.reset_index()


def save_output(df_clean, rfm):
    clean_path = PROC_DIR / "online_retail_clean.csv"
    rfm_path = PROC_DIR / "rfm.csv"

    print(f"\nSaving cleaned dataset to: {clean_path}")
    df_clean.to_csv(clean_path, index=False)

    print(f"Saving RFM table to: {rfm_path}")
    rfm.to_csv(rfm_path, index=False)


if __name__ == "__main__":
    df_raw = load_raw()
    df_clean = clean_data(df_raw)
    rfm = create_rfm(df_clean)
    save_output(df_clean, rfm)
    
