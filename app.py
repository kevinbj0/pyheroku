from flask import Flask, request, jsonify
import requests
import json

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, World!"

@app.route('/api/embeddings', methods=['POST'])
def get_embeddings():
    # Ollama 서버의 ngrok URL
    ngrok_url = "https://32a9-221-146-182-123.ngrok-free.app"  # 예: "http://your-ngrok-url.ngrok.io"

    # API 엔드포인트
    api_endpoint = f"{ngrok_url}/api/embeddings"

    # 요청할 텍스트 데이터
    data = {
        "model": "nomic-embed-text",
        "prompt": "The sky is blue because of Rayleigh scattering"
    }

    # 요청 헤더
    headers = {
        "Content-Type": "application/json"
    }

    # API 요청 보내기
    response = requests.post(api_endpoint, headers=headers, data=json.dumps(data))

    # 응답 확인
    if response.status_code == 200:
        vector = response.json()
        return jsonify(vector)
    else:
        return jsonify({"error": response.status_code, "message": response.text}), response.status_code

if __name__ == '__main__':
    app.run(debug=True)
