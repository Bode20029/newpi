# import cv2
# import torch
# from ultralytics import YOLO
# import os
# import time
# import numpy as np
# import threading
# import queue
# from datetime import datetime
# import pygame
# import random
# from config import *
# from sensors.pzem_sensor import PZEMSensor
# from sensors.distance_sensor import DistanceSensor
# from utils.database_handler import DatabaseHandler
# from utils.line_notifier import LineNotifier

# class CarDetector:
#     def __init__(self):
#         self.model = YOLO(MODEL_PATH)
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         print(f"Using device: {self.device}")
#         self.model.to(self.device)
        
#         self.running = True
#         self.class_colors = {}
#         self.videos = self.load_videos()
        
#         self.distance_sensor_data = queue.Queue(maxsize=5)
#         self.pzem_sensor_data = queue.Queue(maxsize=10)
        
#         self.db_handler = DatabaseHandler()
#         self.line_notifier = LineNotifier(LINE_NOTIFY_TOKEN)
        
#         pygame.mixer.init()
#         self.load_audio_files()

#     def load_audio_files(self):
#         self.audio_files = {
#             "not_charging": pygame.mixer.Sound("sounds/not_charging.mp3"),
#             "alert": pygame.mixer.Sound("sounds/alert.mp3"),
#             "warning": pygame.mixer.Sound("sounds/Warning.mp3")
#         }

#     def load_videos(self):
#         videos_dir = "videos"
#         video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
#         videos = {}
#         for i, file in enumerate(video_files, 1):
#             videos[i] = os.path.join(videos_dir, file)
#         return videos
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
from sensors.pzem_sensor import PZEMSensor
from sensors.distance_sensor import DistanceSensor
from utils.database_handler import DatabaseHandler
from utils.line_notifier import LineNotifier

class CarDetector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        self.running = True
        self.class_colors = {}
        self.videos = self.load_videos()
        
        self.distance_sensor_data = queue.Queue(maxsize=5)
        self.pzem_sensor_data = queue.Queue(maxsize=10)
        
        self.db_handler = DatabaseHandler()
        if not self.db_handler.initialize():
            print("Warning: Database initialization failed. Some features may not work.")
        
        self.line_notifier = LineNotifier(LINE_NOTIFY_TOKEN)
        
        pygame.mixer.init()
        self.load_audio_files()

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

    # def select_video(self):
    #     print("Available videos:")
    #     for i, file in self.videos.items():
    #         print(f"{i}. {os.path.basename(file)}")
        
    #     while True:
    #         try:
    #             choice = int(input("Select a video (enter the number): "))
    #             if choice in self.videos:
    #                 return self.videos[choice]
    #             else:
    #                 print("Invalid choice. Please try again.")
    #         except ValueError:
    #             print("Please enter a valid number.")

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

#     def process_non_ev(self, frame, vehicle_class):
#         print(f"Non-EV car detected: {vehicle_class}")
#         time.sleep(5)  # Wait for 5 seconds
#         self.handle_detection(frame, "Non-EV car parking", vehicle_class)

#     def process_ev(self, frame, vehicle_class):
#         print(f"EV detected: {vehicle_class}")
#         time.sleep(5)  # Wait for 5 seconds
#         print("Starting 10-second charging check")
#         time.sleep(10)  # Simulate 10-second wait
#         self.handle_detection(frame, "EV car not charging", vehicle_class)

#     def handle_detection(self, frame, event, vehicle_class):
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         image_path = f"captures/{event.replace(' ', '_')}_{timestamp}.jpg"
#         cv2.imwrite(image_path, frame)
        
#         file_id = self.db_handler.save_image(frame, timestamp)
#         self.db_handler.save_event(file_id, timestamp, event, vehicle_class)
        
#         message = f"{event} detected at {timestamp}. Vehicle class: {vehicle_class}"
#         self.line_notifier.send_notification(message)
#         self.line_notifier.send_image(message, image_path)
        
#         if "not charging" in event.lower():
#             self.audio_files["not_charging"].play()
#         elif "non-ev" in event.lower():
#             self.audio_files["alert"].play()
#             time.sleep(1)
#             self.audio_files["warning"].play()

#     def run_detection(self):
#         source = self.select_video()
#         cap = cv2.VideoCapture(source)
        
#         if not cap.isOpened():
#             print(f"Error opening video source {source}")
#             return

#         window_name = "Car Detection (Click to stop)"
#         cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#         cv2.setMouseCallback(window_name, lambda event, x, y, flags, param: setattr(self, 'running', False) if event == cv2.EVENT_LBUTTONDOWN else None)

#         frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         cv2.resizeWindow(window_name, frame_width, frame_height)

#         self.simulate_distance_sensor()
#         self.simulate_pzem_sensor()

#         detection_start_time = None
#         detected_class = None

#         while self.running:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             cars = self.detect_cars(frame)
#             frame_with_boxes = self.draw_boxes(frame, cars)

#             if self.distance_sensor_data.full():
#                 distances = list(self.distance_sensor_data.queue)
#                 avg_distance = sum(distances) / len(distances)
#                 if avg_distance < 30 and cars:  # Using 30 cm as threshold
#                     if detection_start_time is None:
#                         detection_start_time = time.time()
#                         detected_class = cars[0][5]  # Get the class of the first detected car
#                     elif time.time() - detection_start_time >= 5:  # 5 seconds of continuous detection
#                         if detected_class in EV_BRANDS:
#                             self.process_ev(frame_with_boxes, detected_class)
#                         else:
#                             self.process_non_ev(frame_with_boxes, detected_class)
#                         self.running = False  # Stop after processing
#                 else:
#                     detection_start_time = None
#                     detected_class = None

#             cv2.imshow(window_name, frame_with_boxes)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()
#         self.db_handler.close()

# def main():
#     detector = CarDetector()
#     detector.run_detection()

# if __name__ == "__main__":
#     main()
    def process_non_ev(self, frame, vehicle_class):
        print(f"Non-EV car detected: {vehicle_class}")
        time.sleep(5)  # Wait for 5 seconds
        self.handle_detection(frame, "Non-EV car parking", vehicle_class)

    def process_ev(self, frame, vehicle_class):
        print(f"EV detected: {vehicle_class}")
        time.sleep(5)  # Wait for 5 seconds
        print("Starting 10-second charging check")
        time.sleep(10)  # Simulate 10-second wait
        self.handle_detection(frame, "EV car not charging", vehicle_class)

    def handle_detection(self, frame, event, vehicle_class):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image_path = f"captures/{event.replace(' ', '_')}_{timestamp}.jpg"
        cv2.imwrite(image_path, frame)
        
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
            self.line_notifier.send_image(message, image_path)
        except Exception as e:
            print(f"Error sending notification: {e}")
        
        if "not charging" in event.lower():
            self.audio_files["not_charging"].play()
        elif "non-ev" in event.lower():
            self.audio_files["alert"].play()
            time.sleep(1)
            self.audio_files["warning"].play()
        
        # Ensure sound is played
        pygame.time.wait(2000)  # Wait for 2 seconds to ensure sound is played

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

        self.simulate_distance_sensor()
        self.simulate_pzem_sensor()

        detection_start_time = None
        detected_class = None

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            cars = self.detect_cars(frame)
            frame_with_boxes = self.draw_boxes(frame, cars)

            if self.distance_sensor_data.full():
                distances = list(self.distance_sensor_data.queue)
                avg_distance = sum(distances) / len(distances)
                if avg_distance < 30 and cars:  # Using 30 cm as threshold
                    if detection_start_time is None:
                        detection_start_time = time.time()
                        detected_class = cars[0][5]  # Get the class of the first detected car
                    elif time.time() - detection_start_time >= 5:  # 5 seconds of continuous detection
                        if detected_class in EV_BRANDS:
                            self.process_ev(frame_with_boxes, detected_class)
                        else:
                            self.process_non_ev(frame_with_boxes, detected_class)
                        self.running = False  # Stop after processing
                else:
                    detection_start_time = None
                    detected_class = None

            cv2.imshow(window_name, frame_with_boxes)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        if self.db_handler:
            self.db_handler.close()

def main():
    detector = CarDetector()
    detector.run_detection()

if __name__ == "__main__":
    main()