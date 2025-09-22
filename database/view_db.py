# view_db.py
import sqlite3

conn = sqlite3.connect("weather_cache.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM weather_cache")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()