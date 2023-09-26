import onnx

model = onnx.load("resnet.onnx")
print(model.opset_import)
