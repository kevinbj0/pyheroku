import os
import pandas as pd
import requests
import json
from flask import Flask




app = Flask(__name__)

@app.route('/')
def hello():
    # CSV 파일 경로
    csv_file_path = os.path.join("data", "초급(전자제품)데이터_최종본.csv")
    
    # CSV 파일 읽기
    df = pd.read_csv(csv_file_path)
    pd.set_option('display.max_rows', None)
    print(df.head())
    return df.head()

if __name__ == "__main__":
    app.run()
