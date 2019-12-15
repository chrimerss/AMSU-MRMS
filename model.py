# model for training
'''
benchmark: pretrained vgg19 without flatten
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

def benchmark():
    model= torchvision.models.vgg19(pretrained=True).features
    model[0]= nn.Conv2d(5,64,3,1,1)
    first= True
    for param in model.parameters():
        if first:
            first=False
        else:
            param.requires_grad= False
    model.add_module(
        nn.Sequential(
            nn.Conv2d(512,64,3,1,1),
            nn.ReLU(True),
            nn.Conv2d(64, 16, 3,1,1),
            nn.ReLU(True),
            nn.Conv2d(16,1,3,1,1),
            nn.ReLU(True)
        )
    )

    return model
