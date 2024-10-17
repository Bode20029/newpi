import cv2
import torch
from ultralytics import YOLO
import time
import os
import RPi.GPIO as GPIO
from datetime import datetime
import pygame
import serial
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu
from config import *
from utils.database_handler import DatabaseHandler
from utils.line_notifier import LineNotifier
import logging


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
                # Set up logging
        logging.basicConfig(filename='pzem_readings.log', level=logging.INFO, 
                            format='%(asctime)s - %(message)s')
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
            return {"voltage": 0, "current_A": 0, "power_W": 0, "energy_Wh": 0, "frequency_Hz": 0, "power_factor": 0}
        try:
            data = self.pzem_sensor.execute(1, cst.READ_INPUT_REGISTERS, 0, 10)
            reading = {
                "voltage": data[0] / 10.0,
                "current_A": (data[1] + (data[2] << 16)) / 1000.0,  # Fixed typo: 'curren t_A' to 'current_A'
                "power_W": (data[3] + (data[4] << 16)) / 10.0,
                "energy_Wh": data[5] + (data[6] << 16),
                "frequency_Hz": data[7] / 10.0,
                "power_factor": data[8] / 100.0
            }
            self.log_pzem_reading(reading)
            return reading
        except Exception as e:
            print(f"Error reading PZEM-004T sensor: {e}")
            return {"voltage": 0, "current_A": 0, "power_W": 0, "energy_Wh": 0, "frequency_Hz": 0, "power_factor": 0}

    def log_pzem_reading(self, reading):
        log_message = (f"Voltage: {reading['voltage']}V, Current: {reading['current_A']}A, "
                    f"Power: {reading['power_W']}W, Energy: {reading['energy_Wh']}Wh, "
                    f"Frequency: {reading['frequency_Hz']}Hz, PF: {reading['power_factor']}")
        logging.info(log_message)
    def detect_cars(self, frame):
        try:
            results = self.model(frame, device=self.device)
            detected_cars = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf)
                    cls = int(box.cls)
                    class_name = self.model.names[cls]
                    
                    # Check if the detected class is in our VEHICLE_CLASSES
                    if class_name in VEHICLE_CLASSES:
                        # Attempt to map the class to a specific EV brand
                        brand = next((brand for brand in EV_BRANDS if brand.lower() in class_name.lower()), "NON-EV")
                        detected_cars.append((x1, y1, x2, y2, conf, brand))
                        print(f"Detected: {brand} with confidence {conf:.2f}, Box: ({x1}, {y1}, {x2}, {y2})")
                        
                        # If we've detected a car, we can break here as we only expect one car
                        break
                
                # Break out of the outer loop as well
                if detected_cars:
                    break
            
            return detected_cars
        except Exception as e:
            print(f"Error in detect_cars: {e}")
            return []
    def draw_boxes(self, frame, cars):
        if cars is None:  # Add a check for None
            return frame
        for car in cars:
            x1, y1, x2, y2, conf, class_name = car
            label = f"{class_name}: {conf:.2f}"

            if class_name in EV_BRANDS:
                color = (0, 255, 0)  # Green for EVs
                label = f"EV - {class_name}: {conf:.2f}"
            else:
                color = (0, 0, 255)  # Red for non-EVs
                label = f"Non-EV - {class_name}: {conf:.2f}"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

    def handle_detection(self, frame, event, vehicle_class):
        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        captures_dir = "captures"
        if not os.path.exists(captures_dir):
            os.makedirs(captures_dir)
        image_path = os.path.join(captures_dir, f"{event.replace(' ', '_')}_{timestamp}.jpg")

        cv2.imwrite(image_path, frame)
        pzem_reading = self.get_pzem_reading()  # Get PZEM reading at detection time
        print(f"Image saved: {image_path}")

        try:
            if self.db_handler.fs:
                file_id = self.db_handler.save_image(frame, timestamp)
                self.db_handler.save_event(file_id, timestamp, event, vehicle_class,pzem_reading)
                print(f"Event saved to database: {event}")
            else:
                print("Warning: Database not initialized. Skipping database operations.")
        except Exception as e:
            print(f"Error saving to database: {e}")

        try:
            message = (f"{event} detected at {timestamp}\n"
           f"Vehicle class: {vehicle_class}\n"
           f"PZEM Reading: Voltage={pzem_reading['voltage']}V, "
           f"Current={pzem_reading['current_A']}A, "
           f"Power={pzem_reading['power_W']}W")
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
                    detected_class = cars[0][5]  # The brand is now in the 5th position
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
                                print(f"EV detected ({current_detection}), checking charging status...")
                                if self.check_charging():
                                    event = f"EV car ({current_detection}) detected (charging)"
                                else:
                                    event = f"EV car ({current_detection}) detected (not charging)"
                            else:
                                event = f"Non-EV car detected"
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

            if cooldown_start and time.time() - cooldown_start >= 10:
                processing = False
                cooldown_start = None
                detection_history.clear()
                current_detection = None

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        detector = QuickEVDetector()
        detector.run_detection()
    finally:
            GPIO.cleanup()