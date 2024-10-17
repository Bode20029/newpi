import hailo_platform
from hailo_platform.tools.hailotools import hef_compiler

def convert_onnx_to_hef(onnx_path, hef_path):
    # Initialize the compiler
    compiler = hef_compiler.HEFCompiler()
    
    # Set compilation options (you may need to adjust these)
    options = {
        "optimize_execution_time": True,
        "target_platform": "hailo8",  # Adjust based on your target hardware
    }
    
    # Compile the ONNX model to HEF
    try:
        compiler.compile(onnx_path, hef_path, options)
        print(f"Successfully converted {onnx_path} to {hef_path}")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

# Usage
onnx_file = "/home/bode/Desktop/ev_monitoring_pi/ev-8s.onnx"
hef_file = "/home/bode/Desktop/ev_monitoring_pi/models/model.hef"
convert_onnx_to_hef(onnx_file, hef_file)