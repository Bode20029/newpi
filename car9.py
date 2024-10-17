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
import random
from config import *
from utils.database_handler import DatabaseHandler
from utils.line_notifier import LineNotifier

class SimulatedSensor(threading.Thread):
    def __init__(self, queue, interval, generate_data_func):
        threading.Thread.__init__(self)
        self.queue = queue
        self.interval = interval
        self.generate_data = generate_data_func
        self.running = True

    def run(self):
        while self.running:
            data = self.generate_data()
            if self.queue.full():
                self.queue.get()  # Remove oldest item if queue is full
            self.queue.put(data)
            time.sleep(self.interval)

    def stop(self):
        self.running = False

class CarDetector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        self.running = True
        self.class_colors = {}
        self.videos = self.load_videos()
        
        self.distance_sensor_data = queue.Queue(maxsize=50)
        self.pzem_sensor_data = queue.Queue(maxsize=100)
        
        self.distance_sensor = SimulatedSensor(
            self.distance_sensor_data, 
            DISTANCE_MEASUREMENT_INTERVAL, 
            self.generate_distance_data
        )
        self.pzem_sensor = SimulatedSensor(
            self.pzem_sensor_data, 
            PZEM_MEASUREMENT_INTERVAL, 
            self.generate_pzem_data
        )
        
        self.db_handler = DatabaseHandler()
        if not self.db_handler.initialize():
            print("Warning: Database initialization failed. Some features may not work.")
        
        self.line_notifier = LineNotifier(LINE_NOTIFY_TOKEN)
        
        pygame.mixer.init()
        self.load_audio_files()
        
        self.alarm_intervals = ALARM_INTERVALS

    def generate_distance_data(self):
        return random.uniform(20, 300)  # Random distance between 20 and 300 cm

    def generate_pzem_data(self):
        return {
            "reading_number": int(time.time()),
            "voltage": random.uniform(220, 240),
            "current_A": random.uniform(0, 1),
            "power_W": random.uniform(0, 240),
        }

    def start_sensors(self):
        self.distance_sensor.start()
        self.pzem_sensor.start()

    def stop_sensors(self):
        self.distance_sensor.stop()
        self.pzem_sensor.stop()
        self.distance_sensor.join()
        self.pzem_sensor.join()

    def get_average_distance(self, num_samples=5):
        distances = []
        for _ in range(num_samples):
            try:
                distances.append(self.distance_sensor_data.get(timeout=1))
            except queue.Empty:
                break
        return sum(distances) / len(distances) if distances else None

    def check_charging(self, duration=10):
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                pzem_data = self.pzem_sensor_data.get(timeout=1)
                if pzem_data['current_A'] > CURRENT_THRESHOLD:
                    return True
            except queue.Empty:
                pass
        return False

    def load_audio_files(self):
        self.audio_files = {
            "not_charging": pygame.mixer.Sound("sounds/not_charging.mp3"),
            "alert": pygame.mixer.Sound("sounds/alert.mp3"),
            "warning": pygame.mixer.Sound("sounds/Warning.mp3")
        }

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
                    selected_video = self.videos[choice]
                    print(f"Selected video: {os.path.basename(selected_video)}")
                    return selected_video
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

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

    def draw_boxes(self, frame, cars):
        for car in cars:
            x1, y1, x2, y2, conf, cls = car
            color = self.get_color_for_class(cls)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

    def get_color_for_class(self, class_name):
        if class_name not in self.class_colors:
            color = tuple(map(int, np.random.randint(0, 255, 3)))
            self.class_colors[class_name] = color
        return self.class_colors[class_name]

    def check_stop_file(self):
        if os.path.exists("stop_detection.txt"):
            os.remove("stop_detection.txt")
            return True
        return False

    def process_ev(self, frame, vehicle_class):
        print(f"EV detected: {vehicle_class}")
        time.sleep(5)  # Wait for 5 seconds to ensure the car is parked
        if self.check_stop_file():
            return
        print("Starting 10-second charging check")
        if not self.check_charging(10):  # 10-second charging check
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

    def handle_detection(self, frame, event, vehicle_class):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        captures_dir = "captures"
        if not os.path.exists(captures_dir):
            os.makedirs(captures_dir)
        image_path = os.path.join(captures_dir, f"{event.replace(' ', '_')}_{timestamp}.jpg")
        
        if frame is None or frame.size == 0:
            print("Error: Empty frame, cannot save image.")
            return

        try:
            cv2.imwrite(image_path, frame)
            print(f"Image saved: {image_path}")
        except Exception as e:
            print(f"Error saving image: {e}")
            return

        try:
            if self.db_handler.fs:
                file_id = self.db_handler.save_image(frame, timestamp)
                self.db_handler.save_event(file_id, timestamp, event, vehicle_class)
            else:
                print("Warning: Database not initialized. Skipping database operations.")
        except Exception as e:
            print(f"Error saving to database: {e}")
        
        try:
            message = f"{event} detected at {timestamp}. Vehicle class: {vehicle_class}"
            self.line_notifier.send_notification(message)
            if os.path.exists(image_path):
                self.line_notifier.send_image(message, image_path)
            else:
                print(f"Warning: Image file not found: {image_path}")
        except Exception as e:
            print(f"Error sending notification: {e}")
        
        self.play_sounds_sequentially(["alert", "warning"] if "non-ev" in event.lower() else ["not_charging"])
        
        print(f"Detection handled for {event}")

    def play_sounds_sequentially(self, sound_names):
        for sound_name in sound_names:
            sound = self.audio_files[sound_name]
            sound.play()
            pygame.time.wait(int(sound.get_length() * 1000))  # Wait for the sound to finish

    def alarm_sequence(self, frame, event, vehicle_class):
        for interval in self.alarm_intervals:
            time.sleep(interval)
            if self.check_stop_file():
                return
            print(f"Vehicle still present, sounding alarm again for: {event}")
            self.handle_detection(frame, event, vehicle_class)

    def simulate_distance_sensor(self):
        for _ in range(5):  # Ensure 5 readings
            distance = random.uniform(20, 29)  # Ensure average is below 30 cm
            self.distance_sensor_data.put(distance)
            time.sleep(DISTANCE_MEASUREMENT_INTERVAL)

    def simulate_pzem_sensor(self):
        for i in range(10):  # 10 seconds of readings
            data = {
                "reading_number": i + 1,
                "voltage": random.uniform(220, 240),
                "current_A": random.uniform(0.05, 0.09),  # Ensure below 0.1 A threshold
                "power_W": random.uniform(10, 20),
            }
            self.pzem_sensor_data.put(data)
            time.sleep(1)  # 1 second interval

    def run_detection(self):
        source = self.select_video()
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error opening video source {source}")
            return

        window_name = "Car Detection (Click to stop)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, lambda event, x, y, flags, param: setattr(self, 'running', False) if event == cv2.EVENT_LBUTTONDOWN else None)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv2.resizeWindow(window_name, frame_width, frame_height)

        self.start_sensors()
        self.simulate_distance_sensor()  # Pre-fill distance sensor queue
        self.simulate_pzem_sensor()  # Pre-fill PZEM sensor queue

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                cars = self.detect_cars(frame)
                frame_with_boxes = self.draw_boxes(frame, cars)

                avg_distance = self.get_average_distance()
                if avg_distance and avg_distance < DISTANCE_THRESHOLD and cars:
                    detected_class = cars[0][5]  # Get the class of the first detected car
                    if detected_class in EV_BRANDS:
                        if not self.check_charging():
                            self.process_ev(frame_with_boxes, detected_class)
                    else:
                        self.process_non_ev(frame_with_boxes, detected_class)

                cv2.imshow(window_name, frame_with_boxes)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.stop_sensors()
            cap.release()
            cv2.destroyAllWindows()
            if self.db_handler:
                self.db_handler.close()

def main():
    detector = CarDetector()
    detector.run_detection()

if __name__ == "__main__":
    main()