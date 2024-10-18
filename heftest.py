# import sys
# print(sys.path)
# import cv2
# import numpy as np
# import hailo_platform.pyhailort as pyhailort




# # Set paths
# hef_file_path = "/home/bode/Desktop/newpi/models/yolov8l.hef"  # Replace with your .hef file path
# camera_id = 0  # Adjust this if needed (0 is usually the default for USB cameras)

# # Initialize Hailo
# def initialize_hailo_runtime(hef_file_path):
#     device = pyhailort.Device.scan_devices()[0]
#     with open(hef_file_path, 'rb') as hef_file:
#         hef_data = hef_file.read()
#     network_group = device.create_network_group('yolo_group', hef_data)
#     network_group.activate()
#     return device, network_group

# # Preprocess frame to match model input size
# def preprocess_frame(frame, input_shape):
#     height, width = input_shape[1:3]
#     resized_frame = cv2.resize(frame, (width, height))
#     return resized_frame

# # Run inference on the camera feed
# def run_inference_on_camera(camera_id, device, network_group):
#     cap = cv2.VideoCapture(camera_id)
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return

#     input_vstream_info = network_group.get_input_vstream_infos()[0]
#     input_shape = input_vstream_info.shape
    
#     print(f"Running inference on camera feed. Press 'q' to quit.")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame.")
#             break
#         # Preprocess the frame
#         preprocessed_frame = preprocess_frame(frame, input_shape)
#         # Convert frame to Hailo input format
#         input_data = np.expand_dims(preprocessed_frame, axis=0).astype(np.uint8)
        
#         # Run inference
#         input_data_dict = {input_vstream_info.name: input_data}
#         output_data_dict = network_group.infer(input_data_dict)
        
#         # Display the output (example: display raw values, modify as per your model)
#         for name, data in output_data_dict.items():
#             print(f"Output {name} shape: {data.shape}")
        
#         # Display the camera feed
#         cv2.imshow("Camera Feed", frame)
#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # Clean up
#     cap.release()
#     cv2.destroyAllWindows()

# # Main function
# def main():
#     # Initialize Hailo runtime and network
#     device, network_group = initialize_hailo_runtime(hef_file_path)
#     # Run inference on the USB camera feed
#     run_inference_on_camera(camera_id, device, network_group)

# if __name__ == "__main__":
#     main()
    
# import hailo_platform.pyhailort as pyhailort
# print(dir(pyhailort))
# print(dir(pyhailort.control_object))
# print(dir(pyhailort.hw_object))
# print(dir(pyhailort.pyhailort))

##################################
# import cv2
# import numpy as np
# import hailo_platform.pyhailort as pyhailort

# # Set paths
# hef_file_path = "/home/bode/Desktop/newpi/models/yolov8l.hef"  # Replace with your .hef file path
# camera_id = 0  # Adjust this if needed (0 is usually the default for USB cameras)

# # Initialize Hailo
# def initialize_hailo_runtime(hef_file_path):
#     # Use VDevice to create the device
#     device = pyhailort.pyhailort.VDevice.create()

#     with open(hef_file_path, 'rb') as hef_file:
#         hef_data = hef_file.read()

#     # Load the HEF file to create the network group
#     network_groups = device.configure(hef_data)
#     network_group = network_groups[0]
#     network_group.activate()

#     return device, network_group

# # Preprocess frame to match model input size
# def preprocess_frame(frame, input_shape):
#     height, width = input_shape[1:3]
#     resized_frame = cv2.resize(frame, (width, height))
#     return resized_frame

# # Run inference on the camera feed
# def run_inference_on_camera(camera_id, device, network_group):
#     cap = cv2.VideoCapture(camera_id)
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return

#     input_vstream_info = network_group.get_input_vstream_infos()[0]
#     input_shape = input_vstream_info.shape
    
#     print(f"Running inference on camera feed. Press 'q' to quit.")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame.")
#             break
#         # Preprocess the frame
#         preprocessed_frame = preprocess_frame(frame, input_shape)
#         # Convert frame to Hailo input format
#         input_data = np.expand_dims(preprocessed_frame, axis=0).astype(np.uint8)
        
#         # Run inference
#         input_data_dict = {input_vstream_info.name: input_data}
#         output_data_dict = network_group.infer(input_data_dict)
        
#         # Display the output (example: display raw values, modify as per your model)
#         for name, data in output_data_dict.items():
#             print(f"Output {name} shape: {data.shape}")
        
#         # Display the camera feed
#         cv2.imshow("Camera Feed", frame)
#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # Clean up
#     cap.release()
#     cv2.destroyAllWindows()

# # Main function
# def main():
#     # Initialize Hailo runtime and network
#     device, network_group = initialize_hailo_runtime(hef_file_path)
#     # Run inference on the USB camera feed
#     run_inference_on_camera(camera_id, device, network_group)

# if __name__ == "__main__":
#     main()

# import hailo_platform.pyhailort as pyhailort

# print(dir(pyhailort.pyhailort.VDevice))

# import cv2
# import numpy as np
# import hailo_platform.pyhailort as pyhailort

# # Set paths
# hef_file_path = "/home/bode/Desktop/newpi/models/yolov8l.hef"  # Replace with your .hef file path
# camera_id = 0  # Adjust this if needed (0 is usually the default for USB cameras)

# # Initialize Hailo
# def initialize_hailo_runtime(hef_file_path):
#     # Open a VDevice
#     vdevice = pyhailort.pyhailort.VDevice()

#     # Get the list of physical devices
#     physical_devices = vdevice.get_physical_devices()
    
#     if not physical_devices:
#         raise RuntimeError("No physical devices found")
    
#     # Load the HEF file into the HEF object
#     hef = pyhailort.HEF(hef_file_path)

#     # Configure the device with the HEF object
#     network_groups = vdevice.configure(hef)
#     network_group = network_groups[0]  # Assuming you want the first network group
#     network_group.activate()

#     return vdevice, network_group

# # Preprocess frame to match model input size
# def preprocess_frame(frame, input_shape):
#     height, width = input_shape[1:3]
#     resized_frame = cv2.resize(frame, (width, height))
#     return resized_frame

# # Run inference on the camera feed
# def run_inference_on_camera(camera_id, device, network_group):
#     cap = cv2.VideoCapture(camera_id)
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return

#     input_vstream_info = network_group.get_input_vstream_infos()[0]
#     input_shape = input_vstream_info.shape
    
#     print(f"Running inference on camera feed. Press 'q' to quit.")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame.")
#             break
#         # Preprocess the frame
#         preprocessed_frame = preprocess_frame(frame, input_shape)
#         # Convert frame to Hailo input format
#         input_data = np.expand_dims(preprocessed_frame, axis=0).astype(np.uint8)
        
#         # Run inference
#         input_data_dict = {input_vstream_info.name: input_data}
#         output_data_dict = network_group.infer(input_data_dict)
        
#         # Display the output (example: display raw values, modify as per your model)
#         for name, data in output_data_dict.items():
#             print(f"Output {name} shape: {data.shape}")
        
#         # Display the camera feed
#         cv2.imshow("Camera Feed", frame)
#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # Clean up
#     cap.release()
#     cv2.destroyAllWindows()

# # Main function
# def main():
#     # Initialize Hailo runtime and network
#     device, network_group = initialize_hailo_runtime(hef_file_path)
#     # Run inference on the USB camera feed
#     run_inference_on_camera(camera_id, device, network_group)

# if __name__ == "__main__":
#     main()
import cv2
import numpy as np
import queue
import hailo_platform.pyhailort as pyhailort
from my_utils import HailoAsyncInference  # Import it from the correct file

# Check if _pyhailort has VDevice and check for available physical devices
def check_physical_devices():
    try:
        # Import VDeviceParams from _pyhailort
        VDeviceParams = pyhailort._pyhailort.VDeviceParams
        
        # Initialize VDeviceParams and explicitly set device_count
        params = VDeviceParams()
        params.device_count = 1  # Ensure it tries to create at least 1 virtual device
        
        # Use the create() method to initialize the VDevice
        vdevice = pyhailort._pyhailort.VDevice.create(params)  # Pass the params
        
        physical_devices = vdevice.get_physical_devices()

        if not physical_devices:
            print("No physical devices found.")
        else:
            print(f"Found physical devices: {physical_devices}")
        return vdevice  # Return the created vdevice for use
    except AttributeError as e:
        print("VDevice class not found:", e)
    except TypeError as e:
        print("Error creating VDevice:", e)
    except pyhailort._pyhailort.HailoRTStatusException as e:
        print(f"Failed to create VDevice: {e}")
    return None

# Initialize Hailo runtime and network group
def initialize_hailo_runtime(hef_file_path):
    try:
        # Import VDeviceParams from _pyhailort
        VDeviceParams = pyhailort._pyhailort.VDeviceParams
        
        # Initialize VDeviceParams and explicitly set device_count
        params = VDeviceParams()
        params.device_count = 1  # Ensure it tries to create at least 1 virtual device
        
        # Use the create() method to initialize the VDevice
        vdevice = pyhailort._pyhailort.VDevice.create(params)  # Pass the params
        
        # Load the HEF file using the HEF class
        hef = pyhailort.HEF(hef_file_path)

        # Configure the device with the HEF object
        network_groups = vdevice.configure(hef)
        network_group = network_groups[0]  # Assuming you want the first network group

        # Activate the network group
        network_group.activate()

        return vdevice, network_group
    except Exception as e:
        print(f"Failed to initialize Hailo runtime: {e}")
        return None, None

# Preprocess frame to match model input size
def preprocess_frame(frame, input_shape):
    height, width = input_shape[1:3]  # Extracting height and width from the input shape
    resized_frame = cv2.resize(frame, (width, height))  # Resize the frame
    return resized_frame

# Run inference on the camera feed
def run_inference_on_camera(camera_id, device, network_group, hef_file_path, batch_size=1):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    input_vstream_info = network_group.get_input_vstream_infos()[0]
    input_shape = input_vstream_info.shape

    # Create input and output queues for async inference
    input_queue = queue.Queue()
    output_queue = queue.Queue()

    print("Initializing async inference...")
    # Initialize HailoAsyncInference for async processing
    inference = HailoAsyncInference(hef_file_path, input_queue, output_queue, batch_size=batch_size)

    print(f"Running inference on camera feed. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame, input_shape)
        
        # Convert frame to Hailo input format
        input_data = np.expand_dims(preprocessed_frame, axis=0).astype(np.uint8)
        
        # Add the preprocessed frame to the input queue for async inference
        print("Adding frame to input queue...")
        input_queue.put(input_data)

        # Check if output is available
        if not output_queue.empty():
            print("Retrieving output from output queue...")
            output_data_dict = output_queue.get()
            # Display the output (modify as per your model requirements)
            for name, data in output_data_dict.items():
                print(f"Output {name} shape: {data.shape}")

        # Display the camera feed
        cv2.imshow("Camera Feed", frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # Define the HEF file path
    hef_file_path = "/home/bode/Desktop/newpi/models/yolov8l.hef"

    # Check for physical devices first
    vdevice = check_physical_devices()

    if vdevice is not None:
        # Initialize Hailo runtime and network
        device, network_group = initialize_hailo_runtime(hef_file_path)
        
        if device is not None and network_group is not None:
            # Run inference on the USB camera feed
            run_inference_on_camera(camera_id=0, device=device, network_group=network_group, hef_file_path=hef_file_path, batch_size=1)
        else:
            print("Failed to initialize the Hailo device.")
    else:
        print("No valid VDevice created.")

if __name__ == "__main__":
    main()
