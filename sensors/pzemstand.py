import time
import serial
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu

class PZEMSensor:
    def __init__(self, port, measurement_interval=5):
        self.port = port
        self.master = None
        self.reading_counter = 0
        self.measurement_interval = measurement_interval

    def connect(self):
        try:
            ser = serial.Serial(
                port=self.port,
                baudrate=9600,
                bytesize=8,
                parity='N',
                stopbits=1,
                xonxoff=0
            )
            self.master = modbus_rtu.RtuMaster(ser)
            self.master.set_timeout(2.0)
            self.master.set_verbose(True)
            print("Successfully connected to PZEM sensor")
            return True
        except Exception as e:
            print(f"Failed to connect to PZEM sensor: {e}")
            return False

    def read_data(self):
        if self.master is None:
            return None
        try:
            data = self.master.execute(1, cst.READ_INPUT_REGISTERS, 0, 10)
            self.reading_counter += 1
            return {
                "reading_number": self.reading_counter,
                "voltage": data[0] / 10.0,
                "current_A": (data[1] + (data[2] << 16)) / 1000.0,
                "power_W": (data[3] + (data[4] << 16)) / 10.0,
                "energy_Wh": data[5] + (data[6] << 16),
                "frequency_Hz": data[7] / 10.0,
                "power_factor": data[8] / 100.0,
                "alarm": data[9]
            }
        except Exception as e:
            print(f"Error reading PZEM data: {e}")
            return None

    def run(self):
        print(f"Starting PZEM sensor readings (Interval: {self.measurement_interval} seconds)")
        while True:
            if self.master is None:
                if not self.connect():
                    time.sleep(5)
                    continue

            data = self.read_data()
            if data is not None:
                print(f"PZEM Reading #{data['reading_number']}:")
                print(f"  Voltage: {data['voltage']}V")
                print(f"  Current: {data['current_A']}A")
                print(f"  Power: {data['power_W']}W")
                print(f"  Energy: {data['energy_Wh']}Wh")
                print(f"  Frequency: {data['frequency_Hz']}Hz")
                print(f"  Power Factor: {data['power_factor']}")
                print(f"  Alarm: {data['alarm']}")
                print("-----------------------------")
            
            time.sleep(self.measurement_interval)

    def close(self):
        if self.master:
            self.master.close()
        print("PZEM sensor connection closed")

def main():
    # Replace '/dev/ttyUSB0' with the appropriate port for your system
    pzem = PZEMSensor('/dev/ttyUSB0', measurement_interval=5)
    try:
        pzem.run()
    except KeyboardInterrupt:
        print("\nStopping PZEM sensor readings...")
    finally:
        pzem.close()

if __name__ == "__main__":
    main()