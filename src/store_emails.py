import sqlite3
import pandas as pd
df = pd.read_csv("/Users/akshajsinha/PycharmProjects/Email_Classification/data/processed/emails_labeled.csv")

connection = sqlite3.connect("emails.db")
cursor = connection.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS
    emails (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        sender TEXT NOT NULL,
        subject TEXT NOT NULL,
        body TEXT NOT NULL,
        gmail_labels TEXT NOT NULL,
        gmail_category TEXT NOT NULL,
        job_candidate TEXT NOT NULL,
        label TEXT NOT NULL
        )
''')

sql_query = "SELECT COUNT(*) FROM emails"
df_from_sql = pd.read_sql_query(sql_query, connection)
print(df_from_sql)
cursor.close()





