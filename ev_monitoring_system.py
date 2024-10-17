import time
import logging
import threading
import queue
import cv2
import pygame
# import Jetson.GPIO as GPIO
import RPi.GPIO as GPIO
from datetime import datetime
from config import *
from sensors.distance_sensor import DistanceSensor
from sensors.pzem_sensor import PZEMSensor
from utils.camera_utils import CameraUtils
from utils.database_handler import DatabaseHandler
from utils.line_notifier import LineNotifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EVMonitoringSystem:
    def __init__(self):
        self.distance_sensor = DistanceSensor(TRIG_PIN, ECHO_PIN)
        self.pzem_sensor = PZEMSensor(PZEM_PORT)
        self.camera_utils = CameraUtils()
        self.db_handler = DatabaseHandler()
        # self.db_handler = DatabaseHandler(MONGO_URI, DB_NAME, COLLECTION_NAME)
        self.line_notifier = LineNotifier(LINE_NOTIFY_TOKEN)
        
        self.distance_sensor_data = queue.Queue(maxsize=5)
        self.pzem_sensor_data = queue.Queue(maxsize=5)
        self.stop_event = threading.Event()
        self.process_active = False
        self.vehicle_present = False

        pygame.mixer.init()

    def initialize_services(self):
        camera_initialized = self.camera_utils.initialize_camera()
        self.camera_utils.initialize_model()  # Add this line
        return (camera_initialized and 
                self.db_handler.initialize() and
                self.line_notifier.initialize())

    def play_sound(self, sound_files):
        for sound_file in sound_files:
            try:
                pygame.mixer.music.load(f"sounds/{sound_file}")
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
            except pygame.error as e:
                logger.error(f"Error playing sound {sound_file}: {e}")

    def handle_detection(self, frame, event, vehicle_class):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if vehicle_class in EV_BRANDS:
            message = f"{vehicle_class} is not charging at {timestamp}"
        else:
            message = f"Non-EV car ({vehicle_class}) parking at {timestamp}"

        if event == "Non-EV car parking":
            self.play_sound(["alert.mp3", "warning.mp3"])
        elif event == "EV car not charging":
            self.play_sound(["not_charging.mp3"])

        try:
            file_id = self.db_handler.save_image(frame, timestamp)
            self.db_handler.save_event(file_id, timestamp, event, vehicle_class)
            logger.info(f"Event data saved successfully with file_id: {file_id}")
        except Exception as e:
            logger.error(f"Failed to save event data: {e}")

        local_path = f"detected_event_{timestamp.replace(':', '-')}.jpg"
        cv2.imwrite(local_path, frame)

        try:
            self.line_notifier.send_image(message, local_path)
            logger.info("Line notification sent successfully")
        except Exception as e:
            logger.error(f"Failed to send Line notification: {e}")

    def monitor_charging(self):
        start_time = time.time()
        last_current = None
        while time.time() - start_time < CHARGING_CHECK_TIME:
            # In your main loop or wherever you process PZEM data
            try:
                pzem_data = self.pzem_sensor_data.get_nowait()
                logger.info(f"PZEM Reading #{pzem_data['reading_number']}: "
                            f"Voltage={pzem_data['voltage']}V, "
                            f"Current={pzem_data['current_A']}A, "
                            f"Power={pzem_data['power_W']}W")
            except queue.Empty:
                pass
            if not self.vehicle_present:
                logger.info("Vehicle no longer present")
                return False
        logger.warning("No charging detected in 10 minutes")
        return False

    def process_ev(self, vehicle_class):
        if self.monitor_charging():
            logger.info(f"{vehicle_class} is charging normally")
        else:
            logger.warning(f"{vehicle_class} not charging")
            ret, frame = self.camera_utils.read_frame()
            if ret:
                self.handle_detection(frame, "EV car not charging", vehicle_class)
            self.alarm_sequence("EV car not charging", vehicle_class)

    def process_non_ev(self, vehicle_class):
        logger.warning(f"Non-EV car detected: {vehicle_class}")
        ret, frame = self.camera_utils.read_frame()
        if ret:
            self.handle_detection(frame, "Non-EV car parking", vehicle_class)
        self.alarm_sequence("Non-EV car parking", vehicle_class)

    def alarm_sequence(self, event, vehicle_class):
        for interval in ALARM_INTERVALS:
            time.sleep(interval)
            if not self.vehicle_present:
                logger.info("Vehicle no longer present")
                return
            logger.warning(f"Vehicle still present, sounding alarm again for: {event}")
            ret, frame = self.camera_utils.read_frame()
            if ret:
                self.handle_detection(frame, event, vehicle_class)

    def run(self):
        if not self.initialize_services():
            logger.error("Failed to initialize services. Exiting.")
            return

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(TRIG_PIN, GPIO.OUT)
        GPIO.setup(ECHO_PIN, GPIO.IN)

        distance_thread = threading.Thread(target=self.distance_sensor.run, args=(self.distance_sensor_data, self.stop_event))
        pzem_thread = threading.Thread(target=self.pzem_sensor.run, args=(self.pzem_sensor_data, self.stop_event))
        display_thread = threading.Thread(target=self.camera_utils.display_thread, args=(self.stop_event,))
        
        distance_thread.start()
        pzem_thread.start()
        display_thread.start()

        logger.info("Starting main detection loop")
        try:
            while not self.stop_event.is_set():
                # Process distance sensor data
                if self.distance_sensor_data.full():
                    distances = list(self.distance_sensor_data.queue)
                    avg_distance = sum(distances) / len(distances)
                    self.vehicle_present = avg_distance < DISTANCE_THRESHOLD
                    logger.info(f"Average Distance: {avg_distance} cm")

                if self.vehicle_present and not self.process_active:
                    self.process_active = True
                    vehicle_detection_start = time.time()
                    stable_detection_count = 0
                    last_detected_class = None

                    while time.time() - vehicle_detection_start < DETECTION_STABILITY_TIME:
                        ret, frame = self.camera_utils.read_frame()
                        if not ret:
                            continue
                        vehicle_present, detected_class = self.camera_utils.detect_vehicle(frame)
                        if not vehicle_present:
                            break
                        if detected_class == last_detected_class:
                            stable_detection_count += 1
                        else:
                            stable_detection_count = 0
                            last_detected_class = detected_class
                        
                        if stable_detection_count >= DETECTION_STABILITY_TIME * 10:  # Assuming 10 FPS
                            if detected_class in EV_BRANDS:
                                threading.Thread(target=self.process_ev, args=(detected_class,)).start()
                            else:
                                threading.Thread(target=self.process_non_ev, args=(detected_class,)).start()
                            break

                    self.process_active = False

                elif not self.vehicle_present and self.process_active:
                    logger.info("Vehicle no longer present")
                    self.process_active = False
                    time.sleep(10)  # Wait 10 seconds before starting next detection cycle

                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Program stopped by user")
        finally:
            self.stop_event.set()
            distance_thread.join()
            pzem_thread.join()
            display_thread.join()
            self.cleanup()

    def cleanup(self):
        self.camera_utils.release()
        self.db_handler.close()
        GPIO.cleanup()
        logger.info("Cleanup completed")