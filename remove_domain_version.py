import onnx

model = onnx.load("resnet.onnx")

# atc 模型转换只支持一个domain_version
if len(model.opset_import) > 1:
    # 删除多余的domain_version
    model.opset_import.pop()

# 将删除domain_version后的模型保存
onnx.save(model, "./resnet.onnx")
