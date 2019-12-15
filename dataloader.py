from netCDF4 import Dataset
from glob import glob
import h5py
import os
import numpy as np


def prepare():
    # initialize two .h5 file, one for training, the other one for testing 7/3
    train_h5= h5py.File('training.h5', 'w')
    test_h5 = h5py.File('testing.h5', 'w')
    # directories to store data
    dirs= ['/Users/hydrosou/Documents/NOAA18/AMSU_GROUND_MERGE_CORRECTED_2', '~/Documents/NOAA19/AMSU_GROUND_MERGE_CORRECTED_2']
    ind= 0 # to construct key
    for each in dirs:
        dataPath= glob(each+'/*.nc') # find all .nc file
        print(dataPath)
        for single in dataPath:
            data= Dataset(single, 'r') #read in netCDF4 object
            lons= data['lon_amsub']    #store longitudes
            lats= data['lat_amsub']    #store latitudes
            mask= np.where((lons[:,0]<=-60) & (lons[:,-1]>=-130) & (lats[:, 0]<=55) & (lats[:, -1]>=25))[0] #mask out US boundary
            if len(mask)!=0:  #if no data observed in US boundary 
                inputChannels= ['c1_amsub', 'c2_amsub', 'c3_amsub', 'c4_amsub', 'c5_amsub', 'aver_precip_nssl'] # input channels, all use amsu-b
                for processedData in preprocess(inputChannels, data, mask):
                    print(processedData.shape)
                    c,m,n= processedData.shape
                    for i in range(0,m-50,20):
                        for j in range(0, n-50, 10):
                            _data= processedData[:,i:i+50, j:j+50]
                            if ind%7!=0:
                                train_h5.create_dataset(str(ind)+'-'+str(i)+'-'+str(j), data=_data)
                            else:
                                test_h5.create_dataset(str(ind)+'-'+str(i)+'-'+str(j), data=_data)
                    #     randomCrop(ind, train_h5, processedData)
                    # elif ind%3!=0:
                    #     randomCrop(ind, test_h5, processedData)
                    ind+=1
                    print('%d/%d'%(ind, len(dataPath)))
    train_h5.close()
    test_h5.close()

def randomCrop(ind, h5, data):
    c,m,n= data.shape
    for i in range(0,m-50,20):
        for j in range(0, n-50, 10):
            _data= data[:,i:i+50, j:j+50]
            h5.create_dataset(str(ind)+'-'+str(i)+'-'+str(j), data=_data)


def preprocess(channels, data, mask):
    print('preprocess...')
    if ((mask[1:]- mask[:-1]>1)).any():
        intermit= np.where((mask[1:] - mask[:-1])>1)[0]
        # print(mask)
        breakPoints= np.concatenate([[0], intermit,[len(mask)]], axis=0)
        # print(breakPoints)
        for i in range(len(breakPoints)-1):
            merged= []
            for channel in channels:
                _data= data[channel][:][mask,:]
                _breakData= _data[breakPoints[i]:breakPoints[i+1]]
                merged.append(_data)
            merged= np.stack(merged)
            yield merged
    else:
        merged= []
        for channel in channels:
            _data= data[channel][:][mask,:]
            merged.append(_data)
        merged= np.stack(merged)

        yield merged


def test():
    h5= h5py.File('testing.h5', 'r')
    print(list(h5.keys()))
    print(len(list(h5.keys())))
    # print(np.array(h5[1]))

if __name__ =='__main__':
    prepare()
    # test()
        

