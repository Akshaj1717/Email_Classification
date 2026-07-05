# train model

import pandas as pd
from pandas import Series
import datetime
from email.header import decode_header

df = pd.read_csv("data/processed/emails_labeled.csv")
 
 # step 1: convert date column to datetime and filter out emails older than 2 years
df["date"] = pd.to_datetime(df["date"], utc=True)

current_time = pd.Timestamp.now(tz="utc")
cutoff_time = current_time - pd.DateOffset(years=2)

df = df[df["date"] >= cutoff_time]
df = df[df["label"] != "Forums"]

df.dropna(subset=["date"], inplace=True)
print(f"Total Emails:", len(df))
print("Label Breakdown:", df["label"].value_counts())

# decoding
def decode_subject(subject):
    if 


# step 2: prepare features and labels for training
df['subject'] = df['subject'].fillna('')
df['body'] = df['body'].fillna('')
df['text'] = "subject: " + df["subject"] + " body: " + df["body"]
print(df["text"].head(3))
