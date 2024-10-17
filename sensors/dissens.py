import os
from gpiozero import DistanceSensor
from gpiozero.pins.lgpio import LGPIOFactory
from time import sleep

# Force gpiozero to use the lgpio pin factory
os.environ['GPIOZERO_PIN_FACTORY'] = 'lgpio'

# Initialize LGPIO pin factory
pin_factory = LGPIOFactory()

class MyDistanceSensor:
    def __init__(self, trig_pin, echo_pin):
        self.sensor = DistanceSensor(trigger=trig_pin, echo=echo_pin, pin_factory=pin_factory)

    def get_distance(self):
        try:
            distance = self.sensor.distance * 100  # Convert to cm
            return round(distance, 2)
        except TimeoutError:
            print("Timeout occurred while measuring distance")
            return None

    def run(self):
        while True:
            dist = self.get_distance()
            if dist is not None:
                print(f"Distance: {dist} cm")
            else:
                print("Failed to measure distance")
            sleep(2)

if __name__ == "__main__":
    sensor = MyDistanceSensor(trig_pin=18, echo_pin=24)  # Adjust pin numbers as needed
    sensor.run()
