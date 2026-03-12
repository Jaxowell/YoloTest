from ultralytics import YOLO
import cv2
import math
import os
import subprocess
import time

model_path = "runs/detect/train6/weights/best.pt" 
total_time = 0
last_frame_time = 0
color = (0, 255, 0)

try:
    model = YOLO(model_path)
except:
    print(f"Ошибка! Нет файла по пути {model_path}")
    exit()

classNames = model.names
print(f"Доступные объекты: {classNames}")

cap = cv2.VideoCapture(1) 
cap.set(3, 1280)
cap.set(4, 720)
total_counts = set()
line_y = 640
steampath = "D:\Steam\steam.exe"

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model.track(frame, persist = True, conf = 0.88, verbose = False)
    cv2.line(frame, (640, 0), (640, 720), (255, 0, 0), 1)

    for r in results:
        gamepad_init = False
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            if box.id != None: track_id = int(box.id[0])
            cls = int(box.cls[0])
            name = classNames[cls]
            centre_x, centre_y = int((max(x1, x2) + min(x1,x2))//2), int((max(y1, y2) + min(y1,y2))//2)
            cv2.circle(frame, (centre_x, centre_y), 2, (120, 120, 0), 3)

            if name == 'gamepad' and box.id != None:
                gamepad_init = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f'{name}. Conf: {conf}. ID: {int(box.id[0])}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                # subprocess.Popen([steampath])
                if (line_y - 20 < centre_x < line_y + 20) and track_id != None:
                    total_counts.add(track_id)
                    cv2.line(frame, (640, 0), (640, 720), (76, 187, 23), 1) 

    if gamepad_init:
        current_time = time.time()
        if last_frame_time != 0:
            delta = current_time - last_frame_time
            total_time += delta
        last_frame_time = current_time
    else: last_frame_time = 0
    
    cv2.putText(frame, f"Общее время игрулек: {int(total_time)}s", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 120, 0), 1)    
    cv2.putText(frame, f"Общее количество прошедших: {len(total_counts)}", (500, 50), cv2.FONT_HERSHEY_COMPLEX, 0.6, (color), 2)

    cv2.imshow("Results", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()