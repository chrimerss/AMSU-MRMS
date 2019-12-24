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
    def __init__(self, type='train'):
        super(DataHelper,self).__init__()
        self.type= type
        if type == 'train':
            h5= h5py.File('training.h5', 'r')
        elif type =='test':
            h5 = h5py.File('testing.h5', 'r')
        elif type=='val':
            h5 = h5py.File('validation.h5', 'r')
        else:
            raise FileNotFoundError('expected type in "train" or "test"!')

        self.keys= list(h5.keys())
        random.shuffle(self.keys)

        h5.close()

    def __getitem__(self, index):
        if self.type == 'train':
            h5= h5py.File('training.h5', 'r')
        elif self.type =='test':
            h5 = h5py.File('/kaggle/input/datausedtotrain/testing.h5', 'r')
        
        key= self.keys[index]
        inputs= np.array(h5[key])[:5,:,:]  #(5,50,50)
        target= np.array(h5[key])[-1,:,:][np.newaxis,:,:]  #(1,50,50)
        inputs[inputs<0]=-1
        target[target<0]=-1
        inputs[np.isnan(inputs)]= -1
        target[np.isnan(target)]= -1
        factor= np.nanmean(target[target>0])/np.nanmean(inputs)
        inputs= inputs*factor
#         print(torch.Tensor(inputs).shape)
#         target= (target - np.nanmean(target[target>0]))/ target[~np.isnan(target)].std()
#         inputs= inputs - inputs.mean(axis=0)
        # normalization

        h5.close()

        return torch.Tensor(inputs), torch.Tensor(target)

    def __repr__(self):
        return 'total samples : %d'%(len(self.keys))

    def __len__(self):
        return len(self.keys)
        