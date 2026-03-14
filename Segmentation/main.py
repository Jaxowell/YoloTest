import cv2
from ultralytics import YOLO
import math
import numpy as np

capture = cv2.VideoCapture(1)

model = YOLO("yolov8n-seg.pt")

capture.set(3, 1280)
capture.set(4, 720)

names = model.names

while True:
    success, frame = capture.read()
    if not success: break

    results = model(frame, stream=True, verbose=False)
    overlay = frame.copy()
    for r in results:
        if r.masks is not None:
            for xy in r.masks.xy:
                mask = np.array((xy), dtype=np.int32)
                overlay = cv2.fillPoly(overlay, [mask], (0, 255, 0), 1)
    frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0.0, 0)

    cv2.imshow("result", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break


capture.release()
cv2.destroyAllWindows()




