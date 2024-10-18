# import hailo_platform.pyhailort as pyhailort

# print(dir(pyhailort))
# import hailo_platform.pyhailort as pyhailort

# print(dir(pyhailort.control_object))
# print(dir(pyhailort.hw_object))
# import hailo_platform.pyhailort as pyhailort

# print(dir(pyhailort.hw_object.PcieDevice))
# print(dir(pyhailort.hw_object.ConfiguredNetwork))
# import hailo_platform.pyhailort as pyhailort

# print(dir(pyhailort.pyhailort.VDevice))
# import hailo_platform.pyhailort as pyhailort

# Get available methods from pyhailort and VDevice
# pyhailort_methods = dir(pyhailort)
# vdevice_methods = dir(pyhailort.pyhailort.VDevice)

# # Write output to text files
# with open("pyhailort_methods.txt", 'w') as f:
#     f.write('\n'.join(pyhailort_methods))

# with open("vdevice_methods.txt", 'w') as f:
#     f.write('\n'.join(vdevice_methods))

# print("Methods saved to pyhailort_methods.txt and vdevice_methods.txt")
# print(dir(pyhailort._pyhailort))

from hailo_sdk_client import ClientRunner

runner = ClientRunner(hw_arch="hailo8l")
hn, npz = runner.translate_onnx_model(onnx_path="model.onnx", model_name="my_model")
runner.save_har("my_model.har")
