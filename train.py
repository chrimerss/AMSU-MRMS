import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from datahelper import DataHelper
from torch.utils.data import DataLoader
from loss import SSIM
import numpy as np
from model import benchmark

USE_GPU= False
LR= 1e-3
BSIZE= 8
EPOCH= 1000

def num_params(net):
    num_params= 0
    for param in net.parameters():
        num_params+= param.numel()

    print('Total number of parameters: %d'%num_params)

def train():
    data= DataHelper(type='test')
    dataLoader= DataLoader(dataset= data, batch_size=BSIZE, shuffle=True)

    #load model
    model= benchmark()
    num_params(model) #print number of parameters

    if USE_GPU:
        model= model.cuda()
    
    criterion = SSIM()
    optimizer= torch.optim.Adam(model.parameters(), lr= LR)
    scheduler= MultiStepLR(optimizer, milestones= [200,400,600], gamma=0.1)

    for epoch in range(EPOCH):
        print('-'*30)
        for param in optimizer.param_groups:
            print('learning rate: ', param['lr'])

        for i, (inputs, target) in enumerate(dataLoader):

            optimizer.zero_grad()
            model.train()

            inputs, target= Variable(inputs), Variable(target)

            if USE_GPU:
                inputs, target= inputs.cuda(), target.cuda()

            out= model(inputs)

            loss= criterion(out, target)

            loss.backward()
            optimizer.step()
            
            print('[%d/%d][%d/%d]  loss: %.4f'%(epoch, EPOCH, i, len(dataLoader), loss.item()))

        scheduler.step()

        if epoch%100 == 0:
            torch.save(model.state_dict, 'model-epoch-%d-benchmark.pth'%epoch)
            

if __name__=='__main__':
    train()