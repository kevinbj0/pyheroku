import requests

url = 'http://192.168.0.11:5000/data'

response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Failed to retrieve data: {response.status_code}")
