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
    train_h5= h5py.File('precipMaskTrain.h5', 'w')
    test_h5 = h5py.File('precipMaskTest.h5', 'w')
    # val_h5 = h5py.File('validating2.h5', 'w')

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

def pixelPrecipType():
    folder= '/Users/hydrosou/Documents/NOAA18/AMSU_GROUND_MERGE_CORRECTED_2'
    maskSurface= np.load('mask.npy')
    files= glob(folder+'/*.nc')
    X_land= []
    y_land= []
    y_land_rr= []
    X_sea= []
    y_sea= []
    y_sea_rr= []
    for k,single in enumerate(files):
        data= Dataset(single, 'r') #read in netCDF4 object
        lons= data['lon_amsub'][:]    #store longitudes
        lats= data['lat_amsub'][:]    #store latitudes
        maskBound= np.where((lons<=-60) & (lons>=-130) & (lats<=55) & (lats>=25)) #mask out US boundary
        # indLats= lats[maskBound]
        # indLons= lons[maskBound]

        # print(np.where(data['aver_precip_nssl'][:][maskBound]>0.1))
        indices= np.where(data['aver_precip_nssl'][:][maskBound]>0.1)[0]

        if len(indices)>0:
            c1_amsua= np.array(data['c1_amsua'][:][maskBound][indices].astype(np.float32))
            c2_amsua= np.array(data['c2_amsua'][:][maskBound][indices].astype(np.float32))
            c15_amsua= np.array(data['c15_amsua'][:][maskBound][indices].astype(np.float32))
            c1_amsub= np.array(data['c1_amsub'][:][maskBound][indices].astype(np.float32))
            c2_amsub= np.array(data['c2_amsub'][:][maskBound][indices].astype(np.float32))
            c3_amsub= np.array(data['c3_amsub'][:][maskBound][indices].astype(np.float32))
            c4_amsub= np.array(data['c4_amsub'][:][maskBound][indices].astype(np.float32))
            c5_amsub= np.array(data['c5_amsub'][:][maskBound][indices].astype(np.float32))
            nsslMask= np.array(data['aver_mask_nssl'][:][maskBound][indices].astype(np.float32))
            rr= np.array(data['aver_precip_nssl'][:][maskBound][indices].astype(np.float32))

            for i, ind in enumerate(indices):
                lon, lat= np.round(lons[maskBound][ind],1), np.round(lats[maskBound][ind],1)
                indRow, indCol= np.where((maskSurface[:,:,0]==lon) & (maskSurface[:,:,1]==lat))

                # print(lon, lat)
                surface= maskSurface[indRow, indCol,2]
                if surface==0:
                    X_sea.append([c1_amsua[i], c2_amsua[i], c15_amsua[i],
                                     c1_amsub[i], c2_amsub[i], c3_amsub[i], c4_amsub[i],
                                      c5_amsub[i]])
                    y_sea.append(nsslMask[i])
                    y_sea_rr.append(rr[i])
                elif surface>0:
                    X_land.append([c1_amsua[i], c2_amsua[i], c15_amsua[i],
                                     c1_amsub[i], c2_amsub[i], c3_amsub[i], c4_amsub[i],
                                      c5_amsub[i]])
                    y_land.append(nsslMask[i])
                    y_land_rr.append(rr[i])

            
        print('%d/%d'%(k, len(files)))


    X_land= np.array(X_land)
    y_land= np.array(y_land)
    X_sea= np.array(X_sea)
    y_sea= np.array(y_sea)
    print(X_land.shape, X_sea.shape, y_land.shape)

    np.save('PrecipTypeX_land.npy',X_land)
    np.save('PrecipTypeY_land.npy',y_land)
    np.save('PrecipRateY_land.npy',y_land_rr)
    np.save('PrecipTypeX_sea.npy',X_sea)
    np.save('PrecipTypeY_sea.npy',y_sea)
    np.save('PrecipRateY_sea.npy',y_sea_rr)




if __name__ =='__main__':
    # prepare()
    pixelPrecipType()
    # test()
        

