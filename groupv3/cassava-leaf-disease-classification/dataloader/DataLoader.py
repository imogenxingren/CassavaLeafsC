import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import rotate, shift
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as dataset


## 归一化 (0,1)标准化
def norm_zero_one(array, span=None):
    '''
    根据所给数组的最大值、最小值，将数组归一化到0-1
    :param array: 数组
    :return: array: numpy格式数组
    '''
    array = np.asarray(array).astype(np.float)
    if span is None:
        mini = array.min()
        maxi = array.max()
    else:
        mini = span[0]
        maxi = span[1]
        array[array < mini] = mini
        array[array > maxi] = maxi

    range = maxi - mini

    def norm(x):
        return (x - mini) / range

    return np.asarray(list(map(norm, array))).astype(np.float)


## 加载一张普通格式图片 2D
def get_normal_image(path, w, h):
    '''
    加载一幅普通格式的2D图像，支持格式：.jpg, .jpeg, .tif ...
    :param path: 医学图像的路径
    :return: array: numpy格式
    '''
    array = Image.open(path).resize((w, h))
    array = np.asarray(array)
    return array


class Dataset(dataset):
    def __init__(self, **config):
        super(Dataset, self).__init__()

        ## 图像文件夹
        self.img_dire = config['data_path']
        ## label路径
        self.csv_path = config["csv_path"]
        ## 图像高、宽
        self.h, self.w = config['h'], config['w']
        ## 增强倍数 [default:1 不增强]
        self.aug_scale = config["aug_scale"]

        print('-----------------------------------------------')
        print('----------- Loading Training Images -----------')
        print('-----------------------------------------------')
        print("Image dire:    {}".format(self.img_dire))
        print("Label path:    {}".format(self.csv_path))
        print("Image w:{}   h:{}".format(self.w, self.h))
        print("Augment scale: {}".format(self.aug_scale))
        time.sleep(0.5)

        ## 只读取csv文件，图片文件索引时读取
        self.csv = pd.read_csv(self.csv_path)
        self.imsize = len(self.csv) * self.aug_scale
        print("Load finished! num:{}".format(self.imsize))

    def __getitem__(self, idx):
        ## 索引
        index = idx // self.aug_scale

        ## 读取图片
        img_id = self.csv.iloc[index, 0]
        img_path = os.path.join(self.img_dire, img_id)
        img = get_normal_image(img_path, self.w, self.h).transpose([2, 0, 1])
        img = norm_zero_one(img, span=[0, 255])  ## 归一化

        ## 因为使用Dataloader封装，不能是字符串，所以这里将XXXX.jpg前的索引提出
        img_id = int(img_id[:-4])

        ## 读取标签
        lab = self.csv.iloc[index, 1]

        ## 进行图像增强
        if index % self.aug_scale != 0:
            random.seed(datetime.now())
            angle = random.uniform(-30, 30)
            ## 因为前面对img的维度进行转换，所以这里旋转的axes换成了(2,1)
            img = rotate(img, angle, axes=(2, 1), reshape=False)

            shifts = [30, 30]
            x_shift = random.uniform(-shifts[0], shifts[0])
            y_shift = random.uniform(-shifts[1], shifts[1])
            img = shift(img, shift=[0, x_shift, y_shift])

            img = norm_zero_one(img, span=[0.0, 1.0])
            img = np.asarray(img).astype(np.float)
            return img_id, img, lab

        img = np.asarray(img).astype(np.float)
        return img_id, img, lab

    def __len__(self):
        return self.imsize


def get_dataloader(**config):
    return DataLoader(dataset=Dataset(**config),
                      batch_size=config["batch_size"],
                      shuffle=config["shuffle"],
                      drop_last=config["drop_last"])


def test_get_dataloader():
    # 可选的数据读取方式
    catalog = {"aug": get_dataloader}

    # 默认的dataloader的参数
    config = {"dataloader": "aug",
              "data_path": "../../Data/cassava-leaf-disease-classification/train_images",
              "csv_path": "../../Data/cassava-leaf-disease-classification/train.csv",
              "aug_scale": 2,
              "batch_size": 32,
              "h": 256,
              "w": 256,
              "shuffle": True,
              "drop_last": True}

    dataloader = catalog[config["dataloader"]](**config)
    import matplotlib.pyplot as plt

    for image_ids, images, labels in dataloader:
        print(len(image_ids))
        print(images.shape)
        print(labels.shape)

        print(image_ids[0])
        image = images.numpy()[0].transpose(1, 2, 0)
        plt.imshow(image)
        plt.show()

        print(labels)

        break


if __name__ == '__main__':
    test_get_dataloader()
