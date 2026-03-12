import cv2
import math
import numpy as np
import requests

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

url = "http://127.0.0.1:8000/predict"

while True:
    ret, frame = cap.read()
    if not ret: break
    success, encoded_image = cv2.imencode('.jpg', frame)

    if success:
        image_bytes = encoded_image.tobytes()
        files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
        response = requests.post(url, files = files, timeout=1)

        data = response.json()

        if data['found_gamepad']:
            for obj in data['all_detections']:
                if obj['confidence'] > 0.9:
                    print("Нашёл:", obj['object'], ". Точность:", round(obj['confidence']*100), "%")
                    coords = obj['box']
                    x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]),
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Result", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()