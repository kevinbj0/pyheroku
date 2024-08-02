import requests

url = 'https://8ca9-221-146-182-45.ngrok-free.app/data'

response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Failed to retrieve data: {response.status_code}")
