import pandas as pd

df = pd.read_csv("data/raw_sales.csv")
df["date"] = pd.to_datetime(df["date"])
df["week"] = df["date"].dt.isocalendar().week.astype(int)
df["month_name"] = df["date"].dt.month_name()
df.to_csv("data/cleaned_sales.csv", index=False)
print(f"Cleaned data saved. Shape: {df.shape}")
print(df.head(3).to_string())
