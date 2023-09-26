import numpy as np
import sys
import os

# 将acllite路径添加到python环境变量
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, ".."))
sys.path.append(os.path.join(path, "../../common"))
sys.path.append(os.path.join(path, "../../common/acllite"))

from acllite_model import AclLiteModel
from acllite_resource import AclLiteResource

MODEL_PATH = "../model/resnet.om"

# init
acl_resource = AclLiteResource()
acl_resource.init()
# load model
model = AclLiteModel(model_path=MODEL_PATH)

test_img = np.load("../data/test_img.npy")
test_img = np.array(test_img)  # (256, 256)
test_img = np.expand_dims(test_img, axis=2)  # (256, 256, 1)
test_img = np.expand_dims(test_img, 0)  # (1, 256, 256, 1)
test_img = test_img.astype(np.float32)

# get output of model
output = model.execute(test_img)

output = np.squeeze(output)
print(output)

class_names = ['CSNJ', 'CW', 'LFM', 'MTJ', 'PBNJ', 'PPNJ']
pred = class_names[output.argmax()]
print(pred)