import cv2
import torch
from ultralytics import YOLO
import os
import time
import numpy as np
import threading
import queue
from datetime import datetime
import pygame
import RPi.GPIO as GPIO
from config import *
from sensors.pzem_sensor import PZEMSensor
from utils.database_handler import DatabaseHandler
from utils.line_notifier import LineNotifier

class CarDetector:
    def __init__(self):
        # ... (keep the existing initialization code)
        self.model = YOLO(MODEL_PATH)
        self.device = self.get_device()
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.running = True
        self.class_colors = {}
        self.videos = self.load_videos()
        
        if self.device == 'cuda':
            self.stream = cv2.cuda_Stream()
        self.alarm_intervals = [10, 20, 30]  # New alarm intervals in seconds
    def get_device(self):
        if not torch.cuda.is_available():
            print("CUDA is not available. Using CPU.")
            return 'cpu'
        
        try:
            cv2.cuda.getCudaEnabledDeviceCount()
        except cv2.error:
            print("OpenCV is not built with CUDA support. Using CPU for OpenCV operations.")
            print("YOLO model will still use GPU if available.")
            return 'cpu'

        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_devices == 0:
            print("No CUDA-capable devices found. Using CPU.")
            return 'cpu'

        print(f"Found {cuda_devices} CUDA-capable device(s).")
        for i in range(cuda_devices):
            props = cv2.cuda.getDevice(i)
            print(f"Device {i}: {props.name()} with compute capability {props.majorVersion()}.{props.minorVersion()}")

        selected_device = cv2.cuda.getDevice()
        print(f"Using CUDA device {selected_device}")
        return 'cuda'

    def enhance_image(self, frame):
        if self.device == 'cuda':
            # Convert to grayscale
            gray = cv2.cuda.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray, self.stream)
            
            # Apply adaptive thresholding
            thresh = cv2.cuda.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2, stream=self.stream)
            
            # Convert back to BGR for consistency with original frame
            enhanced_bgr = cv2.cuda.cvtColor(thresh, cv2.COLOR_GRAY2BGR, stream=self.stream)
        else:
            # CPU fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
            enhanced_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        return enhanced_bgr

    def detect_cars(self, frame):
        # Enhance the frame before detection
        # enhanced_frame = self.enhance_image(frame)
        
        if self.device == 'cuda':
            frame = frame.download()
        
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
            color = tuple(map(int, np.random.randint(0, 255, 3)))
            self.class_colors[class_name] = color
        return self.class_colors[class_name]

    def load_videos(self):
        videos_dir = "videos"
        video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        videos = {}
        for i, file in enumerate(video_files, 1):
            videos[i] = os.path.join(videos_dir, file)
        return videos

    def select_video(self):
        print("Available videos:")
        for i, file in self.videos.items():
            print(f"{i}. {os.path.basename(file)}")
        
        while True:
            try:
                choice = int(input("Select a video (enter the number): "))
                if choice in self.videos:
                    return self.videos[choice]
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def create_stop_button(self, window_name):
        def stop_detection(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.running = False

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, stop_detection)

    def draw_boxes(self, frame, cars):
        if self.device == 'cuda':
            frame_gpu = cv2.cuda_GpuMat(frame)
            for car in cars:
                x1, y1, x2, y2, conf, cls = car
                color = self.get_color_for_class(cls)
                cv2.cuda.rectangle(frame_gpu, (x1, y1), (x2, y2), color, 18, stream=self.stream)
                
                label = f"{cls}: {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                font_thickness = 4
                
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                
                cv2.cuda.rectangle(frame_gpu, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 0), -1, stream=self.stream)
                
                frame_cpu = frame_gpu.download()
                cv2.putText(frame_cpu, label, (x1, y1 - 10), font, font_scale, color, font_thickness)
                frame_gpu.upload(frame_cpu)
            
            frame = frame_gpu.download()
        else:
            for car in cars:
                x1, y1, x2, y2, conf, cls = car
                color = self.get_color_for_class(cls)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 18)
                
                label = f"{cls}: {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                font_thickness = 4
                
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                
                cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 0), -1)
                
                cv2.putText(frame, label, (x1, y1 - 10), font, font_scale, color, font_thickness)
        
        return frame

    def check_stop_file(self):
        if os.path.exists("stop_detection.txt"):
            os.remove("stop_detection.txt")
            return True
        return False
    # ... (keep all the existing methods up to run_detection)

    def process_ev(self, frame, vehicle_class):
        print(f"EV detected: {vehicle_class}")
        time.sleep(5)  # Wait for 5 seconds to ensure the car is parked
        if self.check_stop_file():
            return
        
        charging_detected = self.monitor_charging(5)  # Monitor for 5 seconds
        
        if charging_detected:
            print(f"{vehicle_class} is charging normally")
        else:
            print(f"{vehicle_class} not charging")
            self.handle_detection(frame, "EV car not charging", vehicle_class)
            self.alarm_sequence(frame, "EV car not charging", vehicle_class)

    def process_non_ev(self, frame, vehicle_class):
        print(f"Non-EV car detected: {vehicle_class}")
        time.sleep(5)  # Wait for 5 seconds
        if self.check_stop_file():
            return
        self.handle_detection(frame, "Non-EV car parking", vehicle_class)
        self.alarm_sequence(frame, "Non-EV car parking", vehicle_class)

    def monitor_charging(self, duration):
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                pzem_data = self.pzem_sensor_data.get_nowait()
                print(f"PZEM Reading #{pzem_data['reading_number']}: "
                      f"Voltage={pzem_data['voltage']}V, "
                      f"Current={pzem_data['current_A']}A, "
                      f"Power={pzem_data['power_W']}W")
                if pzem_data['current_A'] > 0.1:
                    return True
            except queue.Empty:
                pass
            if self.check_stop_file():
                return False
        print("No charging detected in the specified time")
        return False

    def alarm_sequence(self, frame, event, vehicle_class):
        for interval in self.alarm_intervals:
            time.sleep(interval)
            if self.check_stop_file():
                return
            print(f"Vehicle still present, sounding alarm again for: {event}")
            self.handle_detection(frame, event, vehicle_class)

    def run_detection(self, timeout=None):
        source = self.select_video()
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error opening video source {source}")
            return

        window_name = "Car Detection (Click to stop)"
        self.create_stop_button(window_name)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cv2.resizeWindow(window_name, frame_width, frame_height)

        start_time = time.time()

        # Start sensor threads
        distance_thread = threading.Thread(target=self.distance_sensor.run, args=(self.distance_sensor_data, self.stop_event))
        pzem_thread = threading.Thread(target=self.pzem_sensor.run, args=(self.pzem_sensor_data, self.stop_event))
        distance_thread.start()
        pzem_thread.start()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            if self.device == 'cuda':
                frame_gpu = cv2.cuda_GpuMat()
                frame_gpu.upload(frame)
                cars = self.detect_cars(frame_gpu)
                frame_with_boxes = self.draw_boxes(frame, cars)
            else:
                cars = self.detect_cars(frame)
                frame_with_boxes = self.draw_boxes(frame, cars)

            cv2.imshow(window_name, frame_with_boxes)
            cv2.waitKey(1)

            # Process distance sensor data
            if self.distance_sensor_data.full():
                distances = list(self.distance_sensor_data.queue)
                avg_distance = sum(distances) / len(distances)
                if avg_distance < DISTANCE_THRESHOLD:
                    for car in cars:
                        _, _, _, _, _, cls = car
                        if cls in EV_BRANDS:
                            threading.Thread(target=self.process_ev, args=(frame, cls)).start()
                        else:
                            threading.Thread(target=self.process_non_ev, args=(frame, cls)).start()

            if self.check_stop_file() or (timeout and time.time() - start_time > timeout):
                break

        self.stop_event.set()
        distance_thread.join()
        pzem_thread.join()
        cap.release()
        cv2.destroyAllWindows()
        self.db_handler.close()

# ... (keep the main function as is)
def main():
    detector = CarDetector()
    timeout = None  # Set to None for no timeout
    detector.run_detection(timeout)

if __name__ == "__main__":
    main()