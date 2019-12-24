

def categoricalError(mask1, mask2):
    '''
    Compute the categorical error:
    contigency tabel
    ---------------------
    sim\obs   rain     no rain        
    rain       a          b
    no rain    c          d
    ---------------------
    POD - a/(a+b)
    FAR - b/(a+b)
    CSI - a/(a+b+c)
    Accuracy - (a+d)/(a+b+c+d)
    frequency bias - (a+b)/(a+c)
    

    '''
