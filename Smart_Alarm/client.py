import requests

url = "http://127.0.0.1:8000/predict"

image_path = "datasets\gamepad\photo_13_2026-02-02_18-16-26.jpg"

files = {
    'file': open(image_path, 'rb')
}

print('Запрос уходит')

repsonse = requests.post(url, files=files)

data = repsonse.json()

if data['found_gamepad']:
    print("ОБЪЕКТ ОБНАРУЖЕН!")
    for obj in data["all_detections"]:
        print("Нашёл:", obj['object'], ". Точность:", round(obj['confidence']*100), "%")

