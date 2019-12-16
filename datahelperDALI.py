'''
By using DALI, the data process step is highly optimized
'''

import os
import sys
import time
import torch
import pickle
import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from sklearn.utils import shuffle
from torchvision.datasets import CIFAR10
from nvidia.dali.pipeline import Pipeline
import torchvision.transforms as transforms
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, dali_cpu=False, local_rank=0,
                 cutout=0):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.iterator = iter(DATA_INPUT_ITER(batch_size, 'train'))
        dali_device = "gpu"
        self.input = ops.ExternalSource()
        self.input_target = ops.ExternalSource()
        

    def iter_setup(self):
        (inputs, targets) = self.iterator.next()
        self.feed_input(self.inputs, inputs)
        self.feed_input(self.targets, targets)

    def define_graph(self):
        self.inputs = self.input()
        self.targets = self.input_target()

        return [self.inputs, self.targets]

class DATA_INPUT_ITER(object):
    
    def __init__(self, batch_size, type='train'):
        self.batch_size = batch_size
        self.train = (type == 'train')
        if self.train:
            self.data= h5py.File('training.h5','r')
        else:
            self.data= h5py.File('testing.h5','r')
        self.keys= list(self.data.keys())
        
    def __iter__(self):
        self.i = 0
        self.n = len(self.keys)
        return self

    def __next__(self):
        batch = []
        targets = []
        for _ in range(self.batch_size):
            inputs, target = self.data[self.keys[self.i]][:5,:,:], self.data[self.keys[self.i]][-1,:,:][np.newaxis,:,:]
            inputs[inputs<0]=-1
            target[target<0]=-1
            inputs[np.isnan(inputs)]= -1
            target[np.isnan(target)]= -1
            factor= np.nanmean(target[target>0])/np.nanmean(inputs)
            inputs= inputs*factor
            inputs[np.isnan(inputs)]= -1
            batch.append(inputs)
            targets.append(target)
            self.i = (self.i + 1) % self.n

        return (batch, targets)

    next = __next__

def get_iter_dali(type, batch_size, num_threads, local_rank=0,  val_size=32, cutout=0):
    if type == 'train':
        pip_train = HybridTrainPipe(batch_size=batch_size, num_threads=num_threads, device_id=local_rank,
                                           local_rank=local_rank, cutout=cutout)
        pip_train.build()
        dali_iter_train = DALIGenericIterator(pip_train, ['inputs', 'target'], size=124160 )
        # dali_iter_train= pip_train.run()
        
        return dali_iter_train