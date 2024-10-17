import cv2
import time
import threading
import torch
from ultralytics import YOLO
from config import MODEL_PATH, VEHICLE_CLASSES

class CameraUtils:
    def __init__(self):
        self.camera = None
        self.yolo_model = None
        self.display_frame = None
        self.display_lock = threading.Lock()

    def initialize_camera(self):
        self.camera = cv2.VideoCapture(0)
        return self.camera.isOpened()

    def initialize_model(self):
        # Check if CUDA is available
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        self.yolo_model = YOLO(MODEL_PATH).to(device)
        print(f"YOLO model loaded on {device}")

    def read_frame(self):
        return self.camera.read()

    def detect_vehicle(self, frame):
        results = self.yolo_model(frame, stream=True)  # stream=True for better performance
        detected = False
        detected_class = None
        for r in results:
            for box in r.boxes:
                class_name = self.yolo_model.names[int(box.cls[0])]
                if class_name in VEHICLE_CLASSES:
                    detected = True
                    detected_class = class_name
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        with self.display_lock:
            self.display_frame = frame.copy()
        
        return detected, detected_class

    def display_thread(self, stop_event):
        while not stop_event.is_set():
            with self.display_lock:
                if self.display_frame is not None:
                    cv2.imshow('EV Monitoring System', self.display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
            
            time.sleep(0.03)  # ~30 FPS

    def release(self):
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()