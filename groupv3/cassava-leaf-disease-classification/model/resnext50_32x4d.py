import pandas as pd
import numpy as np
import torch
from torch import nn
import timm

import random
import os

class resnext50_32x4d(nn.Module):
    def __init__(self, model_name='resnext50_32x4d',pretrained=False, **kwargs):
        super(resnext50_32x4d, self).__init__()
        self.kwargs = kwargs
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = timm.create_model(self.model_name, pretrained=self.pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, self.kwargs["num_classes"])
        
    def forward(self,x):
        y = self.model(x)
        return y