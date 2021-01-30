# Cassava-Leaf-Disease-Classification
kaggle competiton 
## 目录
- 具体参数见模块下的__init__文件

|  模块   | 作用  | 
|  ----  | ----  |
| dataloader  | 数据读取 |
| model  | 模型 |
| trainer  | 训练和验证 |
| train.py  | 模型训练和验证 |
| test.ipynb  | 上交kaggle进行测试 |
| input  | 存放数据集 |
| output  | 训练过程中的输出 |

## 完整流程
### 安装依赖项
- 除pytorch和anaconda已集成的库外还需
```
    pip install timm
```
### 模型训练
- 调整train.py中的config字典
- 执行
```
    python train.py
```
### 上传模型
- 安装kaggle api
    - linux https://blog.csdn.net/fjssharpsword/article/details/105562620
    - windows https://blog.csdn.net/qq_33323162/article/details/82993010
- 新建数据集
```
  kaggle datasets init -p ./output
  # 然后修改dataset-metadata.json中的ID和title参数，ID是数据集名字
  kaggle datasets create -p ./output
``` 
- 更新数据集
```
  kaggle datasets version -p ./output -m "Updated data" 
```

### 获取得分
- 将test.ipynb复制到kaggle的上交jupyter文件中
- 将model部分替换成训练使用的模型  
- 添加刚刚上传的数据集  
- 根据模型修改main模块中的config
- 提交


