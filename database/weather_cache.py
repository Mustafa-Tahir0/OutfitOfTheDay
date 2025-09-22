import sqlite3
import datetime
import requests

DB_NAME = "weather_cache.db"
WEATHER_API_KEY = 'a8677572c8c243fe9ee113017250207'
BASE_URL = 'http://api.weatherapi.com/v1/current.json'

def create_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS weather_cache (
            city TEXT NOT NULL,
            temperature REAL NOT NULL,
            condition TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

create_db()

def get_weather(city):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        SELECT temperature, condition, timestamp FROM weather_cache
        WHERE city = ?
        ORDER BY timestamp DESC
        LIMIT 1
    ''', (city,))
    row = c.fetchone()
    conn.close()

    if row:
        temp, condition, timestamp = row
        timestamp = datetime.datetime.fromisoformat(timestamp)
        if (datetime.datetime.now() - timestamp).total_seconds() < 3600:
            print(f"Using cached weather for {city}")
            return {'temp_f': temp, 'condition': {'text': condition}}

    print(f"Fetching fresh weather for {city}")
    params = {'key': WEATHER_API_KEY, 'q': city}
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    weather = data['current']

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO weather_cache (city, temperature, condition)
        VALUES (?, ?, ?)
    ''', (city, weather['temp_f'], weather['condition']['text']))
    conn.commit()
    conn.close()

    return weather