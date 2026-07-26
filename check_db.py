import sqlite3
import pandas as pd

conn = sqlite3.connect("health.db")

df = pd.read_sql_query("SELECT * FROM history", conn)

print(df)

conn.close()