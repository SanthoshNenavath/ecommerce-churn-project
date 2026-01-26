import pandas as pd

rfm = pd.read_csv("data/processed/rfm.csv")

print("Shape:", rfm.shape)
print("\nHead:")
print(rfm.head())

print("\nDescribe:")
print(rfm[['Recency','Frequency','Monetary']].describe())

print("\nChurn candidates (Recency > 180):")
print((rfm['Recency'] > 180).value_counts())
