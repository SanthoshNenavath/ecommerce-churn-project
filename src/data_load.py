# src/data_load.py
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "online_retail.csv.csv"

def load_raw(path: str = None) -> pd.DataFrame:
    p = RAW if path is None else Path(path)
    print(f"Loading from: {p}")
    df = pd.read_csv(p, encoding="ISO-8859-1")
    return df

def inspect(df: pd.DataFrame, n: int = 5):
    print("Shape:", df.shape)
    print("\nColumns:\n", df.columns.tolist())
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing values (top 10):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))
    print("\nSample rows:")
    print(df.head(n))

if __name__ == "__main__":
    df = load_raw()
    inspect(df, n=10)
