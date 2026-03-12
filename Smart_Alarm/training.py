import cv2
import math
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
if __name__ == '__main__':
    model.train(data = r"datasets\gamepad\Searching gamepad.v2i.yolov8\data.yaml", epochs = 100, imgsz = 640, device = 0)