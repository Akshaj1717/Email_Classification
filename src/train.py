# train model

import pandas as pd
import datetime

df = pd.read_csv("data/processed/emails_labeled.csv")

df["date"] = pd.to_datetime(df["date"], utc=True)
print(df['date'].dtype)
print(df['date'].min(), df['date'].max())

current_time = pd.Timestamp.now(tz="utc")
cutoff_time = current_time - pd.DateOffset(years=2)

df = df[df["date"] >= cutoff_time]

df.dropna(subset=["date"], inplace=True)
print(f"Total Emails:", len(df))
print("Label Breakdown:", df["label"].value_counts())