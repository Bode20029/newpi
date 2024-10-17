import cv2
import torch
from ultralytics import YOLO
from config import MODEL_PATH, VEHICLE_CLASSES
import os
import time
import numpy as np

class CarDetector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.running = True
        self.class_colors = {}

    def detect_cars(self, frame):
        results = self.model(frame, device=self.device)
        detected_cars = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls = int(box.cls)
                class_name = self.model.names[cls]
                if class_name in VEHICLE_CLASSES:
                    detected_cars.append((x1, y1, x2, y2, conf, class_name))
        return detected_cars
    def get_color_for_class(self, class_name):
        if class_name not in self.class_colors:
            # Generate a random color for this class
            color = tuple(map(int, np.random.randint(0, 255, 3)))
            self.class_colors[class_name] = color
        return self.class_colors[class_name]
    # def draw_boxes(self, frame, cars):
    #     for car in cars:
    #         x1, y1, x2, y2, conf, cls = car
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         label = f"{cls}: {conf:.2f}"
    #         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    #     return frame
    # def draw_boxes(self, frame, cars):
        # for car in cars:
        #     x1, y1, x2, y2, conf, cls = car
        #     # Increase thickness of bounding box to 4
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
        #     label = f"{cls}: {conf:.2f}"
        #     # Increase font scale to 1.5 and thickness to 3
        #     (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        #     # Adjust text position
        #     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        # return frame
    def create_stop_button(self, window_name):
        def stop_detection(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.running = False

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, stop_detection)

    def draw_boxes(self, frame, cars):
        for car in cars:
            x1, y1, x2, y2, conf, cls = car
            
            # Get color for this class
            color = self.get_color_for_class(cls)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 18)
            
            label = f"{cls}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            font_thickness = 4
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Create a filled rectangle for text background
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 0), -1)
            
            # Put text on the background
            cv2.putText(frame, label, (x1, y1 - 10), font, font_scale, color, font_thickness)
        
        return frame
    def check_stop_file(self):
        if os.path.exists("stop_detection.txt"):
            os.remove("stop_detection.txt")
            return True
        return False

    def run_detection(self, source, timeout=None):
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error opening video source {source}")
            return

        if self.device == 'cuda':
            cap.set(cv2.CAP_PROP_CUDA_DEVICE, 0)

        window_name = "Car Detection (Click to stop)"
        self.create_stop_button(window_name)
        
        # Get the original video dimensions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set the window size to match the video dimensions
        cv2.resizeWindow(window_name, frame_width, frame_height)

        start_time = time.time()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # Ensure we're working with the original frame size
            original_frame = frame.copy()

            if self.device == 'cuda':
                frame = cv2.cuda_GpuMat(frame)

            cars = self.detect_cars(frame if self.device == 'cpu' else frame.download())

            if self.device == 'cuda':
                frame = frame.download()

            frame_with_boxes = self.draw_boxes(original_frame, cars)

            cv2.imshow(window_name, frame_with_boxes)
            cv2.waitKey(1)  # This is needed to refresh the window

            if self.check_stop_file() or (timeout and time.time() - start_time > timeout):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    detector = CarDetector()
    
    # Replace this with your video file path or camera index
    source = "videos/20240928_135416_1.mp4"  # or 0 for webcam
    
    # Set a timeout (in seconds) or None for no timeout
    timeout = None  # Stop after 60 seconds, or set to None for no timeout
    
    detector.run_detection(source, timeout)

if __name__ == "__main__":
    main()