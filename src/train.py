# train model

import pandas as pd
import datetime

df = pd.read_csv("data/parsed_emails.csv")

df["date"] = pd.to_datetime(df["date"], utc=True)

current_time = pd.Timestamp.now(tz="utc")
cutoff_time = current_time - pd.DateOffset(years=2)

df.dropna(subset=["date"], inplace=True)
print(f"Total Emails:", df.value_counts())
print("Label Breakdown:", df["label"].value_counts())