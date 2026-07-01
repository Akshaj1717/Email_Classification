import pandas as pd
import re

dataset = pd.read_csv("data/processed/emails_labeled.csv")

df = pd.DataFrame(dataset, columns=['date','sender','subject','body','gmail_labels','gmail_category','job_candidate'])
print(df.loc[df['job_candidate'] == True])
print(df['job_candidate'].value_counts())

sender_domain = pd.Series.str.extract(df['sender'], re.search(r'@([A-Za-z0-9.-]+)'))