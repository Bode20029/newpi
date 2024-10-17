# from config import DISTANCE_MEASUREMENT_INTERVAL
# from gpiozero import DistanceSensor
# import time


# class DistanceSensor:
#     def __init__(self, trig_pin, echo_pin):
#         self.sensor = DistanceSensor(trigger=trig_pin, echo=echo_pin)

#     def get_distance(self):
#         try:
#             distance = self.sensor.distance * 100  # Convert to cm
#             return round(distance, 2)
#         except TimeoutError:
#             return None

#     def run(self, queue, stop_event):
#         while not stop_event.is_set():
#             dist = self.get_distance()
#             if dist is not None:
#                 if queue.full():
#                     queue.get_nowait()  # Remove oldest item if queue is full
#                 queue.put_nowait(dist)
#             time.sleep(DISTANCE_MEASUREMENT_INTERVAL)
from config import DISTANCE_MEASUREMENT_INTERVAL
from gpiozero import DistanceSensor as GPIODistanceSensor
import time

class MyDistanceSensor:
    def __init__(self, trig_pin, echo_pin):
        self.sensor = GPIODistanceSensor(trigger=trig_pin, echo=echo_pin)

    def get_distance(self):
        try:
            distance = self.sensor.distance * 100  # Convert to cm
            return round(distance, 2)
        except TimeoutError:
            print("Timeout occurred while measuring distance")
            return None

    def run(self, queue, stop_event):
        try:
            while not stop_event.is_set():
                dist = self.get_distance()
                print(f"Distance measured: {dist}")
                if dist is not None:
                    if queue.full():
                        queue.get_nowait()  # Remove oldest item if queue is full
                    queue.put_nowait(dist)
                time.sleep(DISTANCE_MEASUREMENT_INTERVAL)
        except Exception as e:
            print(f"An error occurred in the distance sensor loop: {e}")

if __name__ == "__main__":
    from multiprocessing import Queue, Event
    
    queue = Queue()
    stop_event = Event()
    
    sensor = MyDistanceSensor(trig_pin=18, echo_pin=24)  # Adjust pin numbers as needed
    sensor.run(queue, stop_event)