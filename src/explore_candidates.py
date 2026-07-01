import pandas as pd
import re

dataset = pd.read_csv("data/processed/emails_labeled.csv")

df = pd.DataFrame(dataset, columns=['date','sender','subject','body','gmail_labels','gmail_category','job_candidate'])


candidates = df.loc[df['job_candidate'] == True]
print(candidates)
print(candidates['job_candidate'].value_counts())

candidates['sender'] = candidates['sender'].str.extract(r'@([A-Za-z0-9.-]+)')

print(candidates['sender'].value_counts().head(15))
print(candidates['sender'].sample(n=15, random_state=42).to_string(index=False))