import sqlite3

# Create (or open) the database
conn = sqlite3.connect("health.db")

cursor = conn.cursor()

# Create History table
cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    age INTEGER,
    bmi REAL,
    glucose REAL,
    blood_pressure REAL,
    skin_thickness REAL,
    insulin REAL,
    pregnancies INTEGER,
    diabetes_pedigree REAL,
    cholesterol REAL,
    heart_rate REAL,
    risk_level INTEGER,
    risk_percentage REAL
)
""")

conn.commit()
conn.close()

print("Database created successfully!")