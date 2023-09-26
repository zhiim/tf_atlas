import ResNet as ResNet
import numpy as np

# 干扰类型
class_names = ['CSNJ', 'CW', 'LFM', 'MTJ', 'PBNJ', 'PPNJ']

# 权重文件的存储路径
model_path = "ResNet.h5"

data_dimension = (256, 256, 1)
model = ResNet.ResNet50(data_dimension)
# 导入网络模型
model.load_weights(model_path)

# 读取测试数据
test_img = np.load('test_img.npy')

# 转换输入数据的维度
test_img = np.array(test_img)  # (256, 256)
test_img = np.expand_dims(test_img, axis=2)  # (256, 256, 1)
test_img = np.expand_dims(test_img, 0)  # (1, 256, 256, 1)

test_output = model.predict(test_img)

# 模型输出的干扰类型
pred = class_names[test_output.argmax()]
print(pred)
