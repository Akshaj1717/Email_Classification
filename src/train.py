# train model


import pandas as pd
from pandas import Series
import datetime
import email
from email.header import decode_header
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

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
    if not pd.notna(subject):
        return subject
    
    try:
        return str(email.header.make_header(decode_header(subject)))
    except Exception:
        return str(subject)
df['subject'] = df['subject'].apply(decode_subject)

# step 2: prepare features and labels for trainin
df['subject'] = df['subject'].fillna('')
df['body'] = df['body'].fillna('')
df['text'] = "subject: " + df["subject"] + " body: " + df["body"]
print(df["text"].head(3))

# step 3: train/test split 
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

# step 4: TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2), sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
