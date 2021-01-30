import os
import cv2
import numpy as np
import torch

# 需要安装第三方库 pip install tfrecord
class TFDataloader:
    def __init__(self, path="./train_tfrecords/", batch_size=32, mode="train"):
        self.mode = mode
        self.batch_size = batch_size
        self.path = path
        self.dataset = self.get_dataset()

    def change_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_dataset(self):
        from tfrecord.torch.dataset import TFRecordDataset
        from itertools import chain

        description = {"image": "byte", "image_name": "byte"}
        if (self.mode == "train"):
            description["target"] = "int"
        index_path = None

        file_list = os.listdir(self.path)
        dataset_list = []
        for i in file_list:
            dataset = TFRecordDataset(self.path + i, index_path, description)
            dataset_list.append(dataset)
        return chain(*dataset_list)

    def __iter__(self):
        return self

    def __next__(self):
        images = None
        labels = None
        image_name = []
        for i in range(self.batch_size):
            try:
                image_dict = next(self.dataset)
            except StopIteration:
                self.dataset = self.get_dataset()
                raise StopIteration

            image_name.append(image_dict["image_name"])
            image = cv2.imdecode(image_dict["image"], cv2.IMREAD_COLOR)
            image = self.pic_processor(image)
            images = self.map_item(image, images)
            label = image_dict["target"]
            labels = self.map_item(label, labels)

        # 将images和labels处理成pytorch可以处理的形式
        images = torch.from_numpy(images).permute(0, 3, 1, 2)
        labels = torch.flatten(labels)
        labels = labels.long()
        labels = torch.from_numpy(labels)
        return images, labels, image_name

    def map_item(self, item, items):
        item = item.reshape((1,) + item.shape)
        if (items is None):
            items = item
        else:
            items = np.concatenate((items, item), axis=0)
        return items

    def pic_processor(self, image):
        image = cv2.resize(image, (227, 227))
        image = image.astype(np.float32) / 255.0
        return image

    def all_image(self):
        images = None
        labels = None
        image_name = []
        self.dataset = self.get_dataset()

        for i in self.dataset:
            image_name.append(i["image_name"])
            image = cv2.imdecode(i["image"], cv2.IMREAD_COLOR)
            image = self.pic_processor(image)
            images = self.map_item(image, images)
            label = i["target"]
            labels = self.map_item(label, labels)

        # 将images和labels处理成pytorch可以处理的形式
        images = torch.from_numpy(images).permute(0, 3, 1, 2)
        labels = torch.flatten(labels)
        labels = labels.long()
        labels = torch.from_numpy(labels)
        return images, labels, image_name