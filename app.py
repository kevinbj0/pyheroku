import requests
import json

# CSV 파일 경로
csv_file_path = os.path.join("data", "초급(전자제품)데이터_최종본.csv")

# CSV 파일 읽기
df = pd.read_csv(csv_file_path)
pd.set_option('display.max_rows', None)
print(df.head())
# Ollama 서버의 ngrok URL
ngrok_url = "https://8c5d-221-146-182-123.ngrok-free.app"  # 예: "http://your-ngrok-url.ngrok.io"

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
    print("Vectorized value:", vector)
else:
    print(f"Error: {response.status_code}, {response.text}")
