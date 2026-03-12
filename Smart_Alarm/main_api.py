from fastapi import FastAPI, UploadFile, File
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

model = YOLO("best.pt")

@app.get("/")
def home():
    return {"message": "YOLO api is Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = model(img)
    detections = []
    gamepad_found = False
    box_coords = None

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            conf = float(box.conf[0])
            box_coords = box.xyxy[0].tolist()
            detections.append({"object": name, "confidence": conf, "box": box_coords})

            if name == 'gamepad':
                gamepad_found = True

    return {"filename": file.filename, "found_gamepad": gamepad_found, "all_detections": detections}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port =8000)