from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov8n-seg.pt')
names = model.names
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, stream=True, verbose=False)

    for r in results:
        if r.masks:
            for mask, box in zip(r.masks.xy, r.boxes):
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) 
                    mask_img = np.zeros_like(frame)

                    if r.masks:
                        for mask, box in zip(r.masks.xy, r.boxes):
                            cnt = mask.astype(np.int32)
           
                            cv2.fillPoly(mask_img, [cnt], (255, 255, 255))

                    final_frame = np.where(mask_img == 255, frame, gray_3ch)
                    cv2.imshow("Sin City Mode", final_frame)

    cv2.imshow("Area Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()