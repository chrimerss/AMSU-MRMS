import numpy as np

def categoricalError(mask1, mask2):
    '''
    Compute the categorical error:
    contigency tabel
    ---------------------
    sim\obs   rain     no rain        
    rain       a          b
    no rain    c          d
    ---------------------
    POD - a/(a+c)
    FAR - b/(a+b)
    CSI - a/(a+b+c)
    Accuracy - (a+d)/(a+b+c+d)
    frequency bias - (a+b)/(a+c)


    '''
    a= ((mask2==1) & (mask1==1)).sum()
    b= ((mask2==0) &  (mask1==1)).sum()
    c= ((mask2==1) & (mask1==0)).sum()
    d= ((mask2==0) & (mask1==0)).sum()
    
    POD= a/(a+c)
    FAR = b/(a+b)
    CSI= a/(a+b+c)
    
    return POD, FAR, CSI

def dice(img1, img2):
    '''
    compute the DICE index

    '''
    img1= np.asarray(img1).astype(np.bool)
    img2= np.asarray(img2).astype(np.bool)
    
    intersection= np.logical_and(img1, img2)
    
    return 2.*intersection.sum()/(img1.sum()+img2.sum())

def computeRMSE(rfModel, segModel, amsuRain):
    '''
    This function computes RMSE with given random forest model and segmentation model

    Args:
    -----------------
    :rfModel - RandomForestRegressor object;
    :segModel - torch.nn.Module object;
    :amsuRain - numpy.array object; rain rate by inherent algorithm

    Output:
    -----------------
    :RMSE_bench - RMSE calculated by AMSU rain rate and reference
    :RMSE_est - RMSE calculated by random forest and reference
    '''
    testhelper= DataRainRate('test')
    RMSEs_bench= []
    RMSEs_est = []
    for ind in range(len(testhelper)):
        print('%d/%d' %(ind,len(testhelper)))
        ins, outs= testhelper[ind]
        outs= outs.numpy().squeeze()
#         ins= ins.numpy().squeeze()
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
        
        amsu= amsuRain[ind]
        RMSEs_est.append(rmse(rains.reshape(-1,1), outs.reshape(-1,1)))
        RMSEs_bench.append(rmse(amsu.reshape(-1,1), outs.reshape(-1,1)))
        
    return RMSEs_est, RMSEs_bench

def rmse(x1, x2):
    '''
    Compute RMSE of two arrays
    '''
    return ((((x1-x2)**2).sum())/len(x1))**0.5