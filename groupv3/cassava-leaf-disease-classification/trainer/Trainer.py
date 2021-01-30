from dataloader import build_dataloader
from model import build_model

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

class Trainer():
    def __init__(self, **config):
        # Load Data
        self.trainloader_config = config['trainloader_config']
        self.validloader_config = config['validloader_config']
        self.trainloader = build_dataloader(**self.trainloader_config)
        self.validloader = build_dataloader(**self.validloader_config)

        # load Model
        self.model_config = config['model_config']
        self.model = build_model(**self.model_config)

        # Set Trainer Config
        train_config = config['train_config']
        self.lr = train_config['lr']
        self.epoch = train_config['epoch']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_func_name = train_config['loss_func']
        self.optimizer_name = train_config['optimizer']
        self.model_path = train_config['model_path']

    def get_loss_func(self, name):
        lossfc_dict = {'CrossEntropy': nn.CrossEntropyLoss()}
        return lossfc_dict.get(name)

    def get_optimizer(self, name):
        optimizer_dict = {'SGD': torch.optim.SGD(self.model.parameters(), lr=self.lr)}
        return optimizer_dict.get(name)

    def run(self):
        print(self.device)
        self.model.to(self.device)
        loss_func = self.get_loss_func(self.loss_func_name)
        optimizer = self.get_optimizer(self.optimizer_name)
        for i in range(self.epoch):
            for j, batch in enumerate(self.trainloader):
                train = batch[1]
                label = batch[2]
                # train = batch['image'] # [batch_size,color,height,weight]
                # label = batch['label'] # [batch]

                optimizer.zero_grad()
                outputs = self.model(train.to(self.device))
                loss = loss_func(outputs, label.to(self.device))
                loss.backward()
                optimizer.step()
                if j % 50 == 0:
                    print(j, loss)
        self.save_model()

    def save_model(self):
        # torch.save(self.model, self.model_path)
        torch.save(self.model.state_dict(), self.model_path)

    def valid(self):
        model = torch.load(self.model_path)
        results = []
        for i, batch in enumerate(self.validloader):
            image = batch[1]
            # image = batch['image']
            pred = model(image.cuda())
            results += torch.max(pred.data, 1)[1].cpu().detach().numpy().tolist()
        self.results = results
        self.metrics(results)
        return results

    def metrics(self, y_pred):
        valid = pd.read_csv(self.validloader_config["csv_path"])
        valid['pred'] = self.results
        acc = (valid['pred'] == valid['label']).sum() / len(valid)
        print('Accuracy in validset is {}'.format(acc))
        report = classification_report(valid['label'], y_pred)
        print(report)