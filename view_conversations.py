import sqlite3

conn = sqlite3.connect("sign_language.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM conversations")
rows = cursor.fetchall()

print("\nSaved Conversations:\n")

for row in rows:
    print(f"ID: {row[0]} | Time: {row[1]} | Sentence: {row[2]}")

conn.close()
