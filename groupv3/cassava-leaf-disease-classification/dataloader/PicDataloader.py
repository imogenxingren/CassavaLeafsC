import os
import cv2
import numpy as np
import torch
import pandas as pd

class PicDataloader:
    def __init__(self, **config):
        self.data_path = config["data_path"]
        self.csv = pd.read_csv(config["csv_path"])
        self.h = config["h"]
        self.w = config["w"]
        self.batch_size = config["batch_size"]
        if(self.batch_size == -1):
            self.batch_size = self.csv.shape[0]
        self.id = 0
        if (config["shuffle"]):
            self.csv = self.csv.sample(frac = 1)

        self.next_tag = 0

    def change_batch_size(self, batch_size):
        if(batch_size == -1):
            self.batch_size = self.csv.shape[0]
        else:
            self.batch_size = batch_size

    def map_item(self, item, items):
        item = item.reshape((1,) + item.shape)
        if (items is None):
            items = item
        else:
            items = np.concatenate((items, item), axis=0)
        return items

    def get_one_data(self, idx):
        img_id = self.csv.iloc[idx, 0]
        img = cv2.imread(os.path.join(self.data_path, self.csv.iloc[idx, 0]))
        img = self.pic_precessor(img)
        label = self.csv.iloc[idx, 1]
        return img_id, img, label

    def __getitem__(self, idx):
        return self.get_one_data(idx)

    def __len__(self):
        return self.csv.shape[0]

    def __iter__(self):
        return self

    def __next__(self):
        if(self.next_tag == 1):
            self.next_tag = 0
            raise StopIteration

        img_ids = []
        imgs = None
        labels = None

        for i in range(self.batch_size):
            if (self.id >= self.csv.shape[0]):
                self.id = 0
                self.csv = self.csv.sample(frac = 1)
                self.next_tag = 1
                break
            else:
                img_id, img, label = self.get_one_data(self.id)
                img_ids.append(img_id)
                imgs = self.map_item(img, imgs)
                labels = self.map_item(label, labels)
                self.id = self.id + 1

        # 将imgs和labels处理成pytorch可以处理的形式
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)
        labels = torch.from_numpy(labels)
        labels = torch.flatten(labels)
        labels = labels.long()
        return img_ids, imgs, labels

    # 图片预处理
    def pic_precessor(self, img):
        img = cv2.resize(img, (self.w, self.h))
        img = img.astype(np.float32) / 255.0
        return img




