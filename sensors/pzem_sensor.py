import time
import serial
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu
from config import PZEM_MEASUREMENT_INTERVAL

class PZEMSensor:
    def __init__(self, port):
        self.port = port
        self.master = None
        self.reading_counter = 0

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

    def run(self, queue, stop_event):
        while not stop_event.is_set():
            if self.master is None:
                if not self.connect():
                    time.sleep(5)
                    continue

            data = self.read_data()
            if data is not None:
                if queue.full():
                    queue.get_nowait()  # Remove oldest item if queue is full
                queue.put_nowait(data)
                print(f"PZEM Reading #{data['reading_number']}: Voltage={data['voltage']}V, Current={data['current_A']}A, Power={data['power_W']}W")
            
            time.sleep(PZEM_MEASUREMENT_INTERVAL)

        if self.master:
            self.master.close()