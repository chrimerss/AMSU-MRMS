'''
Use catalyst framework to infer the resuts
'''
from datahelper import DataHelper
from torch.utils.data import DataLoader
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import InferCallback, CheckpointCallback
import numpy as np
import cv2

def infer(model,type, pth, **kwargs):
    '''
    runs inference for specific model
    Args:
    --------------------
    :model - pytorch model object
    :pth - str; where put the .pth kernel weights
    :kargs - {
        'channels': int (4/8)
        'threshold': int,    #threshold to binarize mask, usually find the optimal with PR AUC curve
        'min_size': int    # how many connected components are considered as real; to filter out noise
        }

    Output:
    --------------------
    :pr_mask - predicted mask according to the order of the validation set
    '''
    channels= kwargs.get('channels', 8)
    test_data= DataHelper(type, channels)
    test_loader= DataLoader(test_data, batch_size=8, shuffle=False)
    
    runner= SupervisedRunner()
    loaders= {"infer": test_loader}
    runner.infer(
            model= model,
            loaders= loaders,
            callbacks=[
            CheckpointCallback(
            resume= pth),
            InferCallback()
                    ]
                )

    probabilities= []
    pr_mask= np.zeros((len(test_data),64,64))

    for i, (batch, output) in enumerate(zip(test_data, runner.callbacks[0].predictions['logits'])):
        # print('%d/%d'%(i, len(probabilities)))
        _, mask= batch #(1,64,64)
        
        for j, probability in enumerate(output):
            probabilities.append(probability)
            pr_mask[i+j,:, :], _= post_process(sigmoid(probability), threshold=kwargs.get('threshold', 0.9), min_size=kwargs.get('min_size', 5))

    return probabilities, pr_mask
            

def post_process(probability, threshold, min_size):
    '''
    post process of segmentation array by using connected objects
    Args:
    ------------------
    :probability - numpy.array; probability map
    :threshold - int; binary thresholding
    :min_size - minimum size of connected objects

    Returns:
    ------------------
    :predictions - numpy.array; refined segmentation map
    :num - int; total number of positive samples

    '''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component= cv2.connectedComponents(mask.astype(np.uint8))
    predictions= np.zeros((64,64), np.float32)
    num=0 
    for c in range(1, num_component):
        p= (component==c)
        if p.sum()>min_size:
            predictions[p]= 1
            num+=1
            
    return predictions, num

def sigmoid(x):
    '''Sigmoid function'''
    return 1/(1+np.exp(-x))


def predictSeg(rfModel, segModel, amsu):
    '''
    Predict rain rate based on random forest regressor and segmentation model

    Args:
    -----------------
    :rfModel - RandomForestRegressor object;
    :segModel - torch.nn.Module object;
    :amsu - numpy array object; 

    Return:
    -----------------
    :rains - numpy.array; predicted rainfall map
    :outs - numpy.array; NSSL reference
    :amsu - numpu.array; benchmark
    '''
    testhelper= DataRainRate('test')
    ind= np.random.randint(len(testhelper))
    ins, outs= testhelper[ind]
    mask= segModel(ins.view(1,8,64,64)).detach().numpy().squeeze()
    mask, _= post_process(mask, threshold=-4.464768, min_size=7)
    rows, cols= np.where(mask==0)
    ins[:, rows, cols]=0
    
    rows, cols= np.where(mask!=0)
    feas= np.zeros((len(rows), 8))
    for l in range(8):
        feas[:,l]= ins[l, rows, cols]
    sims= rfModel.predict(feas)
    rains= np.zeros(outs.shape)
    rains[rows, cols]= sims
    
    return rains, outs, amsu[ind]