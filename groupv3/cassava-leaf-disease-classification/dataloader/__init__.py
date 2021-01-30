from dataloader.PicDataloader import PicDataloader
from dataloader.TFDataloader import TFDataloader

# 可选的数据读取方式
catalog = {"pic": PicDataloader,
           "tf": TFDataloader}

# 默认的dataloader的参数
default_config = {"dataloader": "pic",
                  "data_path": "./input/train_images",
                  "csv_path": "./input/train.csv",
                  "batch_size": 32,
                  "h": 256,
                  "w": 256,
                  "shuffle": True}
'''
Args:
    dataloader代表选择哪个读取器
    data_path代表图片所在路径
    csv_path代表csv文件路径，train为train.csv，test为sample_submission.csv
    batch_size代表一次迭代取出多少
    h,w代表图片预处理后的尺寸
    shuffle代表是否打乱
'''

def build_dataloader(**config):
    default_config.update(config)
    Dataloader = catalog[default_config["dataloader"]]
    return Dataloader(**default_config)

# 使用示例
if __name__ == "__main__":
    dataloader_config = {"data_path": "../input/train_images",
                         "csv_path": "../input/train.csv",}
    dataloader = build_dataloader(**dataloader_config)

    # change_batch_size可更改dataloader的batch_size, -1代表设置为上限
    dataloader.change_batch_size(32)
    dataloader.change_batch_size(-1)

    # dataloader为一个迭代器, 可以直接使用for循环，可以用next方法, 也可以直接索引
    # 返回的格式为image_ids(list), images(tensor), labels(tensor)
    # 1. for循环
    dataloader.change_batch_size(32)
    for image_ids, images, labels in dataloader:
        print(len(image_ids))
        print(images.shape)
        print(labels.shape)

    # 2. next，可用于测试的时候一次性全部取出
    dataloader.change_batch_size(-1)
    image_ids, images, labels = next(dataloader)
    print(len(image_ids))
    print(images.shape)
    print(labels.shape)

    # 3. 直接索引,切片索引没有写，需要的话可以加
    print(len(dataloader))
    image_id, image, label = dataloader[0:64]
    print(image_ids)
    print(image.shape)
    print(label)

