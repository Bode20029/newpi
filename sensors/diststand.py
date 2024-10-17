import time
import RPi.GPIO as GPIO

class HCSR04Sensor:
    def __init__(self, trig_pin, echo_pin, measurement_interval=1):
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.measurement_interval = measurement_interval

        GPIO.setmode(GPIO.BCM)  # Use BCM numbering
        GPIO.setup(self.trig_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)

    def get_distance(self):
        try:
            GPIO.output(self.trig_pin, False)
            time.sleep(0.2)

            GPIO.output(self.trig_pin, True)
            time.sleep(0.00001)
            GPIO.output(self.trig_pin, False)

            while GPIO.input(self.echo_pin) == 0:
                pulse_start = time.time()

            while GPIO.input(self.echo_pin) == 1:
                pulse_end = time.time()

            pulse_duration = pulse_end - pulse_start
            distance = pulse_duration * 17150
            return round(distance, 2)
        except Exception as e:
            print(f"Error measuring distance: {e}")
            return None

    def run(self):
        print(f"Starting HC-SR04 distance measurements (Interval: {self.measurement_interval} seconds)")
        try:
            while True:
                dist = self.get_distance()
                if dist is not None:
                    print(f"Distance: {dist} cm")
                else:
                    print("Error measuring distance")
                time.sleep(self.measurement_interval)
        except KeyboardInterrupt:
            print("\nStopping distance measurements...")
        finally:
            GPIO.cleanup()

# def main():
#     # Use BCM pin numbers
#     TRIG_PIN = 24  # BCM 24, Physical pin 18
#     ECHO_PIN = 23  # BCM 23, Physical pin 16
    
#     sensor = HCSR04Sensor(TRIG_PIN, ECHO_PIN, measurement_interval=1)
#     sensor.run()

# if __name__ == "__main__":
#     main()

# from gpiozero import DistanceSensor
# from time import sleep

# # Define the GPIO pins
# TRIGGER_PIN = 23
# ECHO_PIN = 24

# # Create a distance sensor object
# # Note: we're using the BCM pin numbering
# sensor = DistanceSensor(echo=ECHO_PIN, trigger=TRIGGER_PIN)

# try:
#     while True:
#         # Get the distance in meters
#         distance = sensor.distance
#         # Convert to centimeters
#         distance_cm = distance * 100
#         print(f"Distance: {distance_cm:.1f} cm")
#         sleep(1)

# except KeyboardInterrupt:
#     print("Measurement stopped by User")
#     sensor.close()

# from gpiozero import DistanceSensor
# from gpiozero.pins.pigpio import PiGPIOFactory
# from time import sleep

# # Use pigpio pin factory
# factory = PiGPIOFactory()

# # Define the GPIO pins
# TRIGGER_PIN = 23
# ECHO_PIN = 24

# # Create a distance sensor object
# sensor = DistanceSensor(echo=ECHO_PIN, trigger=TRIGGER_PIN, pin_factory=factory, max_distance=4)

# def read_distance():
#     try:
#         distance = sensor.distance * 100  # Convert to cm
#         return f"Distance: {distance:.1f} cm"
#     except:
#         return "Error reading sensor"

# try:
#     while True:
#         print(read_distance())
#         sleep(1)

# except KeyboardInterrupt:
#     print("Measurement stopped by User")
# finally:
#     sensor.close()