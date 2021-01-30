from model.AlexNet import AlexNet
from model.resnext50_32x4d import resnext50_32x4d
from model.DenseNet import densenet201
from model.SE_resnet import se_resnet152

# 可选的数据读取方式
catalog = {
    "alexnet": AlexNet,
    "resnext50_32x4d":resnext50_32x4d,
    "densenet201":densenet201 
    "se_resnet152":se_resnet152   
    }

# 默认的dataloader的参数
default_config = {"model": "alexnet",
                  "h": 256,
                  "w": 256}
'''
Args:
    model代表选择哪个模型
    h和w代表模型的图片输入大小
'''

def build_model(**config):
    default_config.update(config)
    model = catalog[default_config["model"]]
    return model(**default_config)

