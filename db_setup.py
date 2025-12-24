import sqlite3

conn = sqlite3.connect("sign_language.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    sentence TEXT
)
""")

conn.commit()
conn.close()

print("Database and table created successfully.")
