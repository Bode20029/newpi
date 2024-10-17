# import torch
# print("CUDA Available: ", torch.cuda.is_available())
# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.version.cuda)

# import torch

# def check_cuda_availability():
#     if torch.cuda.is_available():
#         print("CUDA is available.")
#         print(f"Current CUDA device: {torch.cuda.get_device_name(0)}")
#     else:
#         print("CUDA is not available.")

# if __name__ == "__main__":
#     check_cuda_availability()

#################
# import torch
# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"CUDA version: {torch.version.cuda}")

######################
# import cv2
# print(f"OpenCV version: {cv2.__version__}")
# print(f"OpenCV CUDA enabled: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
# if cv2.cuda.getCudaEnabledDeviceCount() > 0:
#     print(f"OpenCV CUDA version: {cv2.cuda.getDevice()}")

# import numpy
# print(numpy.__version__)

# import torch
# import sys

# def test_cuda():
#     print(f"Python version: {sys.version}")
#     print(f"PyTorch version: {torch.__version__}")
#     print(f"CUDA available: {torch.cuda.is_available()}")
#     print(f"CUDA version: {torch.version.cuda}")
#     print(f"CUDA device count: {torch.cuda.device_count()}")
    
#     if torch.cuda.is_available():
#         print(f"Current CUDA device: {torch.cuda.current_device()}")
#         print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
#         # Try a simple CUDA operation
#         x = torch.rand(5, 3)
#         print(f"Input tensor:\n{x}")
#         if torch.cuda.is_available():
#             x = x.cuda()
#             print("Tensor successfully moved to CUDA")
#         print(f"Tensor device: {x.device}")
        
#         # Perform a simple operation
#         y = x * 2
#         print(f"Output tensor (x * 2):\n{y}")
#     else:
#         print("CUDA is not available on this system.")

# if __name__ == "__main__":
#     test_cuda()
import sys
import os

def test_hailo_platform_import():
    try:
        import hailo_platform
        print("Successfully imported hailo_platform module")
        print(f"hailo_platform module location: {hailo_platform.__file__}")
        if hasattr(hailo_platform, '__version__'):
            print(f"hailo_platform module version: {hailo_platform.__version__}")
        else:
            print("hailo_platform module does not have a __version__ attribute")
        print("hailo_platform module attributes:")
        for attr in dir(hailo_platform):
            if not attr.startswith('__'):
                print(f"  - {attr}")
        return True
    except ImportError as e:
        print(f"Failed to import hailo_platform module: {e}")
        return False

def test_hailo_platform_device():
    try:
        import hailo_platform
        device = hailo_platform.Device.create()
        print(f"Successfully created hailo_platform device: {device}")
        device_arch = device.get_architecture()
        print(f"Device architecture: {device_arch}")
        return True
    except Exception as e:
        print(f"Failed to create or query hailo_platform device: {e}")
        return False

def print_environment_info():
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    print(f"sys.path:")
    for path in sys.path:
        print(f"  - {path}")

if __name__ == "__main__":
    print("=== hailo_platform Runtime Test ===")
    print_environment_info()
    print("\nTesting hailo_platform import:")
    if test_hailo_platform_import():
        print("\nTesting hailo_platform device creation:")
        test_hailo_platform_device()
    print("\n=== Test Complete ===")