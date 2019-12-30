import h5py
import torch
import torch.nn as nn
import torch.utils.data as udata
import os
import random
import numpy as np

import h5py
import torch
import torch.nn as nn
import torch.utils.data as udata
import os
import random
import numpy as np

class DataHelper(udata.Dataset):
    def __init__(self, type='train', channels=8):
        super(DataHelper,self).__init__()
        self.type= type
        self.channels= channels
        if type == 'train':
            h5= h5py.File('training2.h5', 'r')
        elif type =='test':
            h5 = h5py.File('testing2.h5', 'r')
        elif type=='val':
            h5= h5py.File('validating2.h5', 'r')
        else:
            raise FileNotFoundError('expected type in "train" or "test"!')


        self.keys= list(h5.keys())
        if type=='train':
            random.shuffle(self.keys)

        h5.close()

    def __getitem__(self, index):
        if self.type == 'train':
            h5= h5py.File('training2.h5', 'r')
        elif self.type =='test':
            h5 = h5py.File('testing2.h5', 'r')
        elif self.type=='val':
            h5= h5py.File('validating2.h5', 'r')
        
        key= self.keys[index]
        if self.channels==4:
            inputs= np.array(h5[key])[4:-2,:,:]
            output= np.array(h5[key])[-2,:,:]
        elif self.channels==8:
            inputs= np.array(h5[key])[:-2,:,:]  #(9,50,50)
            output= np.array(h5[key])[-2,:,:] #(1,50,50)
        
        # output[output<0]=np.nan 
        # imputer = KNNImputer(n_neighbors=4, weights="uniform")
        # output= imputer.fit_transform(output)
        assert output.shape==(64,64), print(output.shape)
        mask= np.zeros((1,64,64))
        mask[0,:,:][output>0]= 1
#         mask[1,:,:][output==0]= 0
#         mask[2,:,:][output<0]= -1
        for i in range(inputs.shape[0]):
            #normalize to (-1,1)
            # print(inputs[i,:,:])
            if inputs[i,:,:].max()>0:
                nanmin= (inputs[i,:,:][inputs[i,:,:]>0]).min()
#                 inputs[i,:,:][inputs[i,:,:]>0]= -(inputs[i,:,:][inputs[i,:,:]>0]-inputs[i,:,:].max())/(inputs[i,:,:].max()-nanmin)
                inputs[i,:,:][output<0]=0
                inputs[i,:,:][inputs[i,:,:]<0]=0
            else:
                inputs[i,:,:]= 0
        # mask= np.array(output>0).sum(axis=0)[np.newaxis, :,:]
        
        return [torch.Tensor(inputs), torch.Tensor(mask)]

    def __repr__(self):
        return 'total samples : %d'%(len(self.keys))

    def __len__(self):
        return len(self.keys)
    
    def val_data(self):
        h5= h5py.File('validating2.h5', 'r')
        amsuRR= np.zeros((len(self.keys), 64,64))
        for i,key in enumerate(self.keys):
            _amsuRR= np.array(h5[key])[-1,:,:]
            _amsuRR[_amsuRR>0]=1
            _amsuRR[_amsuRR<=0]=0
            amsuRR[i]= _amsuRR
            
        return amsuRR