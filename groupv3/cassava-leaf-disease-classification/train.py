import os
import pandas as pd
from sklearn.model_selection import train_test_split

from trainer import Trainer

def file_split(**config):
    if not os.path.isdir("./output"):
        os.mkdir("./output")
    train_file = config['train_file']
    mytrain_file = config['mytrain_file']
    valid_file = config['valid_file']
    test_size = config['test_size']
    data = pd.read_csv(train_file)
    train, valid = train_test_split(data, test_size=test_size, random_state=0)
    train.to_csv(mytrain_file, index=False)
    valid.to_csv(valid_file, index=False)
    print('Split Done')

config = {
    "split_config": {
        'train_file': './input/train.csv',
        'mytrain_file': './output/mytrain.csv',
        'valid_file': './output/valid.csv',
        'test_size': 0.25
    },

    "trainloader_config": {
        "data_path": "./input/train_images/",
        "csv_path": "./output/mytrain.csv",
        "batch_size": 32,
        "h": 256,
        "w": 256,
        "shuffle": True
    },

    "validloader_config": {
        "data_path": "./input/train_images/",
        "csv_path": "./output/valid.csv",
        "batch_size": 32,
        "h": 256,
        "w": 256,
        "shuffle": False
    },

    "model_config": {
        "model": "resnext50_32x4d",
        "h": 256,
        "w": 256,
        "num_classes": 5
    },

    "train_config": {
        'lr': 0.02,
        'epoch': 20,
        'loss_func': 'CrossEntropy',
        'optimizer': 'SGD',
        'model_path': './output/resnet.pt'}
}

'''
Args:(split_config)
    train_file代表原始训练集csv文件路径
    mytrain_file代表新训练集csv文件路径
    valid_file代表验证集csv文件路径
    test_size代表训练集和验证集的划分比例
'''

# 划分训练集和验证集,在output文件夹下面生成两个新的csv，mytrain.csv和valid.csv
file_split(**config['split_config'])

# 训练,在output文件中生成pkl模型文件
train = Trainer(**config)
train.run()

# 验证，结果会写在valid.csv中
train.valid()


