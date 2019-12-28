'''
Use catalyst framework to infer the resuts
'''
from datahelper import DataHelper
from torch.utils.data import DataLoader
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import InferCallback, CheckpointCallback
import numpy as np
import cv2

def infer(model, pth, **kwargs):
    '''
    runs inference for specific model
    Args:
    --------------------
    :model - pytorch model object
    :pth - str; where put the .pth kernel weights
    :kargs - {
        'threshold': int,    #threshold to binarize mask, usually find the optimal with PR AUC curve
        'n_components': int    # how many connected components are considered as real; to filter out noise
        }

    Output:
    --------------------
    :pr_mask - predicted mask according to the order of the validation set
    '''

    test_data= DataHelper('val')
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

    valid_masks= []
    pr_mask= np.zeros((len(test_data),64,64))

    for i, (batch, output) in enumerate(zip(test_data, runner.callbacks[0].predictions['logits'])):
        print('%d/%d'%(i, len(probabilities)))
        _, mask= batch #(1,64,64)
        for m in mask:
            valid_masks.append(m)
        
        for j, probability in enumerate(output):
            pr_mask[i+j,:, :], _= post_process(probability, **kwargs)

    return pr_mask
            

def post_process(probability, threshold, min_size):
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
    return 1/(1+np.exp(-x))
