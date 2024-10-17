import cv2
import torch
from ultralytics import YOLO
import time
import os
from datetime import datetime, timezone
import pytz
import pygame
import serial
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu
from config import *
from utils.database_handler import DatabaseHandler
from utils.line_notifier import LineNotifier

class QuickEVDetector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        self.db_handler = DatabaseHandler()
        if not self.db_handler.initialize():
            print("Warning: Database initialization failed. Some features may not work.")

        self.line_notifier = LineNotifier(LINE_NOTIFY_TOKEN)

        pygame.mixer.init()
        self.load_audio_files()

        self.pzem_sensor = self.initialize_pzem_sensor()

    def load_audio_files(self):
        self.audio_files = {
            "not_charging": pygame.mixer.Sound("sounds/not_charging.mp3"),
            "alert": pygame.mixer.Sound("sounds/alert.mp3"),
            "warning": pygame.mixer.Sound("sounds/Warning.mp3")
        }

    def initialize_pzem_sensor(self):
        try:
            ser = serial.Serial(
                port='/dev/ttyUSB0',  # Update this to match your PZEM-004T connection
                baudrate=9600,
                bytesize=8,
                parity='N',
                stopbits=1,
                timeout=1
            )
            master = modbus_rtu.RtuMaster(ser)
            master.set_timeout(2.0)
            master.set_verbose(True)
            print("PZEM-004T sensor initialized successfully")
            return master
        except Exception as e:
            print(f"Failed to initialize PZEM-004T sensor: {e}")
            return None

    def get_pzem_reading(self):
        if self.pzem_sensor is None:
            return {"current_A": 0}
        try:
            data = self.pzem_sensor.execute(1, cst.READ_INPUT_REGISTERS, 0, 10)
            current_A = (data[1] + (data[2] << 16)) / 1000.0
            return {"current_A": current_A}
        except Exception as e:
            print(f"Error reading PZEM-004T sensor: {e}")
            return {"current_A": 0}

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
            color = (0, 255, 0) if cls in EV_BRANDS else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

    def handle_detection(self, frame, event, vehicle_class):
        thailand_tz = pytz.timezone('Asia/Bangkok')
        timestamp = datetime.now(timezone.utc).astimezone(thailand_tz).strftime("%d-%m-%Y %H:%M:%S")
        captures_dir = "captures"
        if not os.path.exists(captures_dir):
            os.makedirs(captures_dir)
        image_path = os.path.join(captures_dir, f"{event.replace(' ', '_')}_{timestamp}.jpg")
        
        cv2.imwrite(image_path, frame)
        print(f"Image saved: {image_path}")

        try:
            if self.db_handler.fs:
                file_id = self.db_handler.save_image(frame, timestamp)
                self.db_handler.save_event(file_id, timestamp, event, vehicle_class)
                print(f"Event saved to database: {event}")
            else:
                print("Warning: Database not initialized. Skipping database operations.")
        except Exception as e:
            print(f"Error saving to database: {e}")
        
        try:
            message = f"{event} detected at {timestamp}\nVehicle class: {vehicle_class}"
            self.line_notifier.send_notification(message)
            self.line_notifier.send_image(message, image_path)
            print(f"Notification sent: {message}")
        except Exception as e:
            print(f"Error sending notification: {e}")
        
        print(f"Playing audio for: {event}")
        if "EV" in event and "not charging" in event.lower():
            print("Playing 'not_charging' audio for EV not charging")
            self.audio_files["not_charging"].play()
            pygame.time.wait(int(self.audio_files["not_charging"].get_length() * 1000))
        elif "Non-EV" in event:
            print("Playing 'alert' audio for Non-EV")
            self.audio_files["alert"].play()
            pygame.time.wait(int(self.audio_files["alert"].get_length() * 1000))
            print("Playing 'warning' audio for Non-EV")
            self.audio_files["warning"].play()
            pygame.time.wait(int(self.audio_files["warning"].get_length() * 1000))
        else:
            print(f"No specific audio defined for event: {event}")

        print(f"Audio playback completed for: {event}")
        pygame.mixer.stop()

    def check_charging(self, duration=5):
        start_time = time.time()
        initial_current = self.get_pzem_reading()["current_A"]
        while time.time() - start_time < duration:
            current_reading = self.get_pzem_reading()["current_A"]
            if abs(current_reading - initial_current) >= 0.1:
                return True
            time.sleep(0.1)
        return False

    def run_detection(self):
        cap = cv2.VideoCapture(0)  # Use 0 for default webcam
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        window_name = "EV Detection (Press 'q' to quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        detection_start_time = None
        current_detection = None
        processing = False
        cooldown_start = None
        detection_history = []
        max_history = 10  # Keep track of last 10 detections

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break

            if not processing and cooldown_start is None:
                cars = self.detect_cars(frame)
                frame_with_boxes = self.draw_boxes(frame, cars)

                if cars:
                    detected_class = cars[0][5]
                    detection_history.append(detected_class)
                    if len(detection_history) > max_history:
                        detection_history.pop(0)
                    
                    most_common_detection = max(set(detection_history), key=detection_history.count)
                    
                    if most_common_detection != current_detection:
                        current_detection = most_common_detection
                        detection_start_time = time.time()
                        print(f"New detection: {current_detection}")
                    elif current_detection:
                        detection_duration = time.time() - detection_start_time
                        required_duration = 5 if current_detection in EV_BRANDS else 3
                        stable_detections = detection_history.count(current_detection)
                        
                        if detection_duration >= required_duration and stable_detections >= required_duration:
                            processing = True
                            if current_detection in EV_BRANDS:
                                print("EV detected, checking charging status...")
                                if self.check_charging():
                                    event = "EV car detected (charging)"
                                else:
                                    event = "EV car detected (not charging)"
                            else:
                                event = "Non-EV car detected"
                            self.handle_detection(frame_with_boxes, event, current_detection)
                            cooldown_start = time.time()
                else:
                    detection_history.clear()
                    current_detection = None
                    detection_start_time = None

            else:
                frame_with_boxes = frame  # During processing or cooldown, use the original frame

            cv2.imshow(window_name, frame_with_boxes)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Check if cooldown period is over
            if cooldown_start and time.time() - cooldown_start >= 10:
                processing = False
                cooldown_start = None
                current_detection = None
                detection_start_time = None
                detection_history.clear()
                print("Cooldown period over. Resuming detection.")

        cap.release()
        cv2.destroyAllWindows()
        if self.db_handler:
            self.db_handler.close()
        if self.pzem_sensor:
            self.pzem_sensor.close()
        
def main():
    detector = QuickEVDetector()
    detector.run_detection()

if __name__ == "__main__":
    main()

    