from netCDF4 import Dataset
from glob import glob
import h5py
import os
import numpy as np


def prepare():
    '''
    Output three data types
    each has (13,64,64)
    (0,64,64): c1_amsua
    (1,64,64): c2_amsua
    (2,64,64): c15_amsua
    (3,64,64): c1_amsub
    (4,64,64): c2_amsub
    (5,64,64): c3_amsub
    (6,64,64): c4_amsub
    (7,64,64): c5_amsub
    (8,64,64): straitiform rain
    (9,64,64): snow
    (10,64,64): convective rain
    (11,64,64): rain rate by NSSL
    (12,64,64): rain rate by AMSU
    '''
    # initialize two .h5 file, one for training, the other two for validation and testing 7/2/1
    train_h5= h5py.File('training2.h5', 'w')
    test_h5 = h5py.File('testing2.h5', 'w')
    val_h5 = h5py.File('validating2.h5', 'w')

    # directories to store data
    dirs= ['/Users/hydrosou/Documents/NOAA18/AMSU_GROUND_MERGE_CORRECTED_2', '~/Documents/NOAA19/AMSU_GROUND_MERGE_CORRECTED_2']
    ind= 0 # to construct key
    for each in dirs:
        dataPath= glob(each+'/*.nc') # find all .nc file
        # print(dataPath)
        for single in dataPath:
            data= Dataset(single, 'r') #read in netCDF4 object
            lons= data['lon_amsub']    #store longitudes
            lats= data['lat_amsub']    #store latitudes
            mask= np.where((lons[:,0]<=-60) & (lons[:,-1]>=-130) & (lats[:, 0]<=55) & (lats[:, -1]>=25))[0] #mask out US boundary
            if len(mask)!=0:  #if no data observed in US boundary 
                inputChannels= ['c1_amsua', 'c2_amsua','c15_amsua','c1_amsub', 'c2_amsub',
                 'c3_amsub', 'c4_amsub', 'c5_amsub','aver_precip_nssl', 'rr_amsub'] # input channels, all use amsu-b
                for processedData in preprocess(inputChannels, data, mask):
                    target= processedData[-2,:,:]
                    randLayer= np.random.randint(8)
                    if np.nanmean(target[target>0])>1:# constrain record rainy samples
                        print(processedData.shape)
                        c,m,n= processedData.shape
                        for i in range(0,m-64,20):
                            for j in range(0, n-64, 5):
                                _data= processedData[:,i:i+64, j:j+64]
                                if (_data[randLayer,:,:]>0).any() and np.nanmean(_data[-2,:,:][_data[-2,:,:]>0])>1 and np.nanmean(_data[-1,:,:][_data[-1,:,:]>0])>0.5:
                                    if ind%7!=0:
                                        train_h5.create_dataset(str(ind)+'-'+str(i)+'-'+str(j), data=_data)
                                    elif ind%2==0:
                                        val_h5.create_dataset(str(ind)+'-'+str(i)+'-'+str(j), data=_data)
                                    else:
                                        test_h5.create_dataset(str(ind)+'-'+str(i)+'-'+str(j), data=_data)
                # except IndexError:
                #     pass
                        #     randomCrop(ind, train_h5, processedData)
                        # elif ind%3!=0:
                        #     randomCrop(ind, test_h5, processedData)
                    ind+=1
                    
                    print('%d/%d'%(ind, len(dataPath)))
    train_h5.close()
    test_h5.close()

def randomCrop(ind, h5, data):
    c,m,n= data.shape
    for i in range(0,m-64,20):
        for j in range(0, n-64, 5):
            _data= data[:,i:i+50, j:j+50]
            h5.create_dataset(str(ind)+'-'+str(i)+'-'+str(j), data=_data)


def preprocess(channels, data, mask):
    # print('preprocess...')
    if ((mask[1:]- mask[:-1]>1)).any():
        intermit= np.where((mask[1:] - mask[:-1])>1)[0]
        # print(mask)
        breakPoints= np.concatenate([[0], intermit,[len(mask)]], axis=0)
        # print(breakPoints)
        for i in range(len(breakPoints)-1):
            merged= []
            for channel in channels:
                if channel=='aver_mask_nssl':

                    if data[channel][:].shape[0]==10:
                        strait= data[channel][:][1:3,:,:][mask,:][breakPoints[i]:breakPoints[i+1]].sum(axis=0)
                        snow= data[channel][:][3,:,:][mask,:][breakPoints[i]:breakPoints[i+1]]
                        convec= data[channel][:][6,:,:][mask,:][breakPoints[i]:breakPoints[i+1]]
                        strait[strait>=5]=1
                        strait[strait<5]=0
                        merged.append(strait)
                        snow[snow>=5]=1
                        snow[snow<5]=0
                        merged.append(snow)
                        convec[convec>=5]=1
                        convec[convec<5]=0
                        merged.append(convec)
                        
                    elif data[channel][:].shape[-1]==10:
                        strait= data[channel][:][:,:,1:3][mask,:][breakPoints[i]:breakPoints[i+1]].sum(axis=0)
                        snow= data[channel][:][:,:,3][mask,:][breakPoints[i]:breakPoints[i+1]]
                        convec= data[channel][:][:,:,6][mask,:][breakPoints[i]:breakPoints[i+1]]
                        strait[strait>=5]=1
                        strait[strait<5]=0
                        merged.append(strait)
                        snow[snow>=5]=1
                        snow[snow<5]=0
                        merged.append(snow)
                        convec[convec>=5]=1
                        convec[convec<5]=0
                        merged.append(convec)
                else:
                    _data= data[channel][:][mask,:][breakPoints[i]:breakPoints[i+1]]
                merged.append(_data)
            merged= np.stack(merged)
            yield merged
    else:
        merged= []
        for channel in channels:
            if channel=='aver_mask_nssl':
                    if data[channel][:].shape[0]==10:
                        strait= data[channel][:][1:3,:,:][mask,:].sum(axis=0)
                        snow= data[channel][:][3,:,:][mask,:]
                        convec= data[channel][:][6,:,:][mask,:]
                        strait[strait>=5]=1
                        strait[strait<5]=0
                        merged.append(strait)
                        snow[snow>=5]=1
                        snow[snow<5]=0
                        merged.append(snow)
                        convec[convec>=5]=1
                        convec[convec<5]=0
                        merged.append(convec)
                    elif data[channel][:].shape[-1]==10:
                        strait= data[channel][:][:,:,1:3][mask,:].sum(axis=0)
                        snow= data[channel][:][:,:,3][mask,:]
                        convec= data[channel][:][:,:,6][mask,:]
                        strait[strait>=5]=1
                        strait[strait<5]=0
                        merged.append(strait)
                        snow[snow>=5]=1
                        snow[snow<5]=0
                        merged.append(snow)
                        convec[convec>=5]=1
                        convec[convec<5]=0
                        merged.append(convec)
            else:
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
        

