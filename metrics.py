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

