import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = 'postgresql://postgres:1234@<공인 IP>:5432/koreanBeef'

def test_db_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        print(f"Connected to PostgreSQL Database - Version: {db_version}")
        cursor.close()
        conn.close()
    except Exception as error:
        print(f"Error connecting to PostgreSQL Database: {error}")

if __name__ == "__main__":
    test_db_connection()
