import requests

url = "http://127.0.0.1:8000/predict"

sample_input = {
    "Temperature": 20.5,
    "Humidity": 27.2,
    "Light": 200,
    "CO2": 900,
    "HumidityRatio": 0.0065
}

response = requests.post(url, json=sample_input)

print("Status Code:", response.status_code)
print("Response JSON:")
print(response.json())
