import pandas as pd
import re

dataset = pd.read_csv("data/processed/emails_labeled.csv")

df = pd.DataFrame(dataset, columns=['date','sender','subject','body','gmail_labels','gmail_category','job_candidate'])
print(df.loc[df['job_candidate'] == True])
print(df['job_candidate'].value_counts())

df['sender'] = df['sender'].str.extract(r'@([A-Za-z0-9.-]+)')

print(df['sender'].value_counts().head(15))
print(df.sample(n=15, random_state=42).to_string(index=False))