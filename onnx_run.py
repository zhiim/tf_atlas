import onnxruntime as onnxrt
import numpy as np

class_names = ['CSNJ', 'CW', 'LFM', 'MTJ', 'PBNJ', 'PPNJ']

# load model
model = onnxrt.InferenceSession("resnet.onnx",
                providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
print("loading model success!")

# input data
test_img = np.load('test_img.npy')
test_img = np.array(test_img, dtype=np.float32)  # (256, 256)
test_img = np.expand_dims(test_img, axis=2)  # (256, 256, 1)
test_img = np.expand_dims(test_img, 0)  # (1, 256, 256, 1)

# input
sf_input = {model.get_inputs()[0].name:test_img}

# output
output = model.run(None, sf_input)
print("get output of spatial filter success!")
output = np.array(output)

pred = class_names[output.argmax()]
print(pred)
