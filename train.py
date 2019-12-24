'''
using DALI batch_size= 256 costs 4.06 minutes for one epoch
without DALI, it costs 12 minutes
'''
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
from unet import UNet
from datahelperDALI import get_iter_dali


USE_GPU= True
LR= 1e-3
BSIZE= 32
EPOCH= 100

def init_weights(m):
    classname= m.__class__.__name__
    
    if classname.find('conv2d')!=-1:
        m.weight.data.uniform(0.0,1.0)
        m.bias.data.fill_(0.0)

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
    
#     criterion = SSIM()
    criterion= nn.MSELoss()
    optimizer= torch.optim.Adam(model.parameters(), lr= LR)
    scheduler= MultiStepLR(optimizer, milestones= [20,40,60], gamma=0.1)


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
                criterion= criterion.cuda()

            out= model(inputs)
            
            
            loss= criterion(out[target>0], target[target>0])
            
            acc= nn.MSELoss()(out, target).item()

            loss.backward()
            optimizer.step()
            
            print('[%d/%d][%d/%d]  loss: %.4f Spatial loss: %.4f'%(epoch, EPOCH, i, len(dataLoader), loss.item(), acc))

        scheduler.step()

        if epoch%50 == 0:
            torch.save(model.state_dict, 'model-epoch-%d-benchmark.pth'%epoch)
            
def trainDALI():
    '''
    training with DALI as data loader
    '''
    model= UNet()
    model.apply(init_weights)
    num_params(model) #print number of parameters

    if USE_GPU:
        model= model.cuda()
    
#     criterion = SSIM()
    criterion= nn.MSELoss()
    ssim= SSIM()
    optimizer= torch.optim.Adam(model.parameters(), lr= LR)
    scheduler= MultiStepLR(optimizer, milestones= [20,40,60], gamma=0.1)
    writer= SummaryWriter('./log')
    

    for epoch in range(EPOCH):
        print('-'*30)
        start= time.time()
        for param in optimizer.param_groups:
            print('learning rate: ', param['lr'])
        train_loader = get_iter_dali(type='train', batch_size=256,
                                        num_threads=8)
        for i, data in enumerate(train_loader):
            data= data[0]

            optimizer.zero_grad()
            model.train()

            inputs, target= Variable(data['inputs']).to(dtype=torch.float), Variable(data['target']).to(dtype=torch.float)
            
            if USE_GPU:
                inputs, target= inputs.cuda(), target.cuda()
                criterion= criterion.cuda()

            out= model(inputs)
            
            
            loss= criterion(out[target>0], target[target>0])
            
            acc= ssim(out, target)

            loss.backward()
            optimizer.step()
            
            print('[%d/%d][%d/%d]  loss: %.4f Spatial accuracy: %.4f'%(epoch, EPOCH, i, 485, loss.item(), acc))
            if i%10 ==0:
                writer.add_scalar('loss', loss.item(), i+1)
                writer.add_scalar('accuracy', acc, i+1)

            if i% 100 ==0:
                model.eval()
                out= model(inputs)
                # add image to tensorboard
                batch_num= np.random.randint(0,BSIZE)
                sim= out[batch_num,0,:,:]
                sim= utils.make_grid(sim,normalize=True, scale_each=True, nrow=8)
                target= utils.make_grid(target.squeeze()[batch_num,:,:],normalize=False, scale_each=True, nrow=8)

                writer.add_image('model', sim, i+1)
                writer.add_image('target', target, i+1)


        scheduler.step()

        if (epoch+1)%10 == 0:
            torch.save(model.state_dict, 'model-epoch-%d-benchmark-trainset2.pth'%epoch)
        
        end= time.time()
        print('training one epoch elapses %.2f minutes!'%((end-start)/60.))

if __name__=='__main__':
    train()