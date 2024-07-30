from flask import Flask
import os
import psycopg2
from psycopg2.extras import RealDictCursor

app = Flask(__name__)

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:1234@<공인 IP>:5432/koreanBeef')

def get_db_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

@app.route('/')
def hello():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute('SELECT version();')
        db_version = cursor.fetchone()
        cursor.close()
        conn.close()
        return f"Connected to PostgreSQL Database - Version: {db_version}"
    else:
        return "Failed to connect to database"

if __name__ == '__main__':
    app.run()
