import cv2
import numpy as np
from ultralytics import YOLO

mouse_point = (0, 0)

model = YOLO("yolov8n-seg.pt")
class_names = model.names

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)
zone_points = np.array([[350, 500], [200, 100], [600, 100]])
zone_points = zone_points.reshape((-1, 1, 2))

while True:
    ret, frame = cap.read()
    if not ret: break
    is_safe = True
    color = (0, 255, 0)
    status = "Safe"
    detected_center = None
    track_id = None
    results = model.track(frame, persist=True, verbose = False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            name_cls = class_names[cls]
            centre_x, centre_y = int((x1 + x2) // 2), int((y1 + y2) // 2)
            result_inside = cv2.pointPolygonTest(zone_points, (centre_x, centre_y), False)
            if result_inside >= 0 and name_cls == "person":
                color = (0, 0, 255)
                status = "Zone!"
                is_safe = False
                detected_center = centre_x, centre_y
                if box.id is not None:
                    track_id = int(box.id[0])
                break

    cv2.polylines(frame, [zone_points], isClosed=True, color=color, thickness=3)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone_points], color)

    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    if not is_safe and detected_center != None:
        cv2.circle(frame, (detected_center), 5, (255, 255, 255), -1)
        cv2.putText(frame, f"Intruder ID: {track_id}", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("result", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
