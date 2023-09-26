import tensorflow as tf
import ResNet as ResNet
import os

model_path = 'ResNet.h5'  # 权重文件路径

data_dimension = (256, 256, 1)
model = ResNet.ResNet50(data_dimension)

# 读取网络模型
model.load_weights(model_path)

# export model to savedmodel
mobilenet_save_path = os.path.join("./model")
tf.saved_model.save(model, mobilenet_save_path)
