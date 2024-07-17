import os
import pandas as pd
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello():
    # CSV 파일 경로
    csv_file_path = os.path.join(os.path.dirname(__file__), "data", "초급(전자제품)데이터_최종본.csv")
    
    try:
        # CSV 파일 읽기
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        pd.set_option('display.max_rows', None)
        return df.head().to_json(orient='records', force_ascii=False)
    except FileNotFoundError:
        return jsonify({"error": f"CSV file not found at {csv_file_path}"}), 404

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
