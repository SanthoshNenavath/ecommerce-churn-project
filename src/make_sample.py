import pandas as pd

# Load the full raw dataset
df = pd.read_csv("data/raw/online_retail.csv", encoding="ISO-8859-1")

# Create a 1k sample
sample = df.sample(1000, random_state=42)

# Save the sample to data/raw
sample.to_csv("data/raw/sample_online_retail_1k.csv", index=False)

print("Sample CSV created at: data/raw/sample_online_retail_1k.csv")
