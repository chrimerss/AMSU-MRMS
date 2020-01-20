from netCDF4 import Dataset
from glob import glob
import h5py
import os
import numpy as np
from affine import Affine
from osgeo import gdal
import multiprocessing
from functools import partial


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
    (9,64,64): convective
    (10,64,64): snow
    (11,64,64): rain rate by NSSL
    (12,64,64): rain rate by AMSU
    '''
    # initialize two .h5 file, one for training, the other two for validation and testing 7/2/1
    train_h5= h5py.File('precipMaskTrain.h5', 'w')
    test_h5 = h5py.File('precipMaskTest.h5', 'w')
    # val_h5 = h5py.File('validating2.h5', 'w')

    # directories to store data
    dirs= ['/Users/hydrosou/Documents/NOAA18/AMSU_GROUND_MERGE_CORRECTED_2']
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
                 'c3_amsub', 'c4_amsub', 'c5_amsub','aver_mask_nssl', 'aver_precip_nssl', 'rr_amsub'] # input channels, all use amsu-b
                for processedData in preprocess(inputChannels, data, mask):
                    if processedData.shape[1]>64 and processedData.shape[2]>64: #image size larger than 64
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
                                        # elif ind%2==0:
                                        #     val_h5.create_dataset(str(ind)+'-'+str(i)+'-'+str(j), data=_data)
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
                    
                    strait= np.array(data[channel][:][:,:,1:3][mask,:]).sum(axis=2)[breakPoints[i]:breakPoints[i+1]]
                    snow= np.array(data[channel][:][:,:,3][mask,:][breakPoints[i]:breakPoints[i+1]])
                    convec= np.array(data[channel][:][:,:,6][mask,:][breakPoints[i]:breakPoints[i+1]])

                    nrows, ncols= strait.shape
                    maskType= np.zeros((3, nrows, ncols), dtype=np.int16)
                    for m in range(nrows):
                        for n in range(ncols):
                            if snow[m,n]>=10: maskType[-1,m,n]=1
                            elif convec[m,n]>=10: maskType[-2,m,n]=1
                            elif strait[m,n]>=10: maskType[-3,m,n]=1
                        
                    merged.append(maskType[0,:,:])
                    merged.append(maskType[1,:,:])
                    merged.append(maskType[2,:,:])
                else:
                    # print(np.array(data[channel][:]).shape, breakPoints, breakPoints)
                    _data= np.array(data[channel][:][mask,:])[breakPoints[i]:breakPoints[i+1]]
                    merged.append(_data)
            merged= np.stack(merged)
            print(merged.shape)
            yield merged
    else:
        merged= []
        for channel in channels:
            if channel=='aver_mask_nssl':
                strait= np.array(data[channel][:][:,:,1:3][mask,:].sum(axis=2))
                snow= np.array(data[channel][:][:,:,3][mask,:])
                convec= np.array(data[channel][:][:,:,6][mask,:])
                nrows, ncols= data[channel][:][:,:,6][mask,:].shape
                maskType= np.zeros((3,nrows, ncols), dtype=np.int16)
                for m in range(nrows):
                    for n in range(ncols):
                        if snow[m,n]>=10: maskType[-1,m,n]=1
                        elif convec[m,n]>=10: maskType[-2,m,n]=1
                        elif strait[m,n]>=10: maskType[-3,m,n]=1
                        

                merged.append(maskType[0,:,:])
                merged.append(maskType[1,:,:])
                merged.append(maskType[2,:,:])
            else:
                _data= np.array(data[channel][:][mask,:])
                merged.append(_data)
        merged= np.stack(merged)
        print(merged.shape)

        yield merged


def test():
    h5= h5py.File('testing.h5', 'r')
    print(list(h5.keys()))
    print(len(list(h5.keys())))
    # print(np.array(h5[1]))

def pixelPrecipData(threads):
    '''
    Make PrecipTypeY.npy including 10 classes
         PrecipRate.npy including nssl rain rates
         PrecipTypeX.npy including 12 features: 8 channels local pixels plus four non-local features (add lons, lats, surface type, dem)
    '''

    folder= '/Users/hydrosou/Documents/NOAA18/AMSU_GROUND_MERGE_CORRECTED_2'
    maskSurface= np.load('mask.npy')
    files= glob(folder+'/*.nc')[:5]
    dem= gdal.Open('geotiffs/global_elevation.tif')
    dem_arr= dem.ReadAsArray()
    transform= Affine.from_gdal(*dem.GetGeoTransform())
    X= []
    ycls= []
    yrr= []
    

    with multiprocessing.Pool(threads) as pool:
        iter_func= partial(thread, maskSurface, transform, dem_arr)
        results= pool.map(iter_func, files)

    for (fea, precipClass, rainrate) in results:
        if len(fea)>0:
            X.append(fea)
            ycls.append(precipClass)
            yrr.append(rainrate)

    X= np.concatenate(X)
    ycls= np.concatenate(ycls)
    yrr= np.concatenate(yrr)

    print(X.shape, ycls.shape, yrr.shape)

    np.save('PrecipPointsX.npy',X)
    np.save('PrecipPointsTypeY.npy',ycls)
    np.save('PrecipPointsRateY.npy',yrr)

def thread( maskSurface, transform, dem_arr, single):
    print('processing %s'%single)
    X= []
    ycls= []
    yrr= []
    data= Dataset(single, 'r') #read in netCDF4 object
    lons= data['lon_amsub'][:]    #store longitudes
    lats= data['lat_amsub'][:]    #store latitudes
    maskBound= np.where((lons<=-60) & (lons>=-130) & (lats<=60) & (lats>=25)) #mask out US boundary
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
            indAMSURow, indAMSUCol= np.where((np.round(lons,1)==lon) & (np.round(lats,1)==lat))
            demCol, demRow= ~transform*(lon, lat)
            demPoint= dem_arr[int(demRow), int(demCol)]
            # print(indASMURow, indASMUCol)
            if len(indAMSURow)>1 and len(indAMSUCol)>1:
                indAMSURow= indAMSURow[0]
                indAMSUCol= indAMSUCol[0]
            if 89>indAMSUCol>0:
                # print(indCOl)
                VI, VC, VX, VM= makeNeighboringData(data, indAMSURow, indAMSUCol)
                # print(lon, lat)
                surface= maskSurface[indRow, indCol, 2].astype(float) 
                surface= np.nan if len(surface)==0 else surface
                # print(surface)
                X.append([c1_amsua[i], c2_amsua[i], c15_amsua[i],
                                        c1_amsub[i], c2_amsub[i], c3_amsub[i], c4_amsub[i],
                                        c5_amsub[i], VI, VC, VX, VM, surface, demPoint, lon, lat])
                ycls.append(nsslMask[i])
                yrr.append(rr[i])

    return np.array(X).astype(float), np.array(ycls).astype(float), np.array(yrr).astype(float)


def makeNeighboringData(data, row, col):
    c5= np.array(data['c15_amsua'][:])
    c1= np.array(data['c1_amsua'][:])
    c2= np.array(data['c2_amsua'][:])
    c4= np.array(data['c1_amsub'][:])
    
    VI= 1/8*(abs(c5[row, col]-c5[row+1, col])+abs(c5[row, col]-c5[row-1, col])+
                                 abs(c5[row, col]-c5[row, col-1])+abs(c5[row, col]-c5[row, col+1])+
                                abs(c5[row, col]-c5[row+1, col+1])+abs(c5[row, col]-c5[row-1, col-1])+
                                abs(c5[row, col]-c5[row+1, col-1])+abs(c5[row, col]-c5[row-1, col+1]))

    VC= c1[row, col]- 1/8*(c1[row+1, col]+c1[row-1, col]+c1[row, col+1]+c1[row, col-1]+
                                c1[row+1, col+1]+c1[row-1, col-1]+c1[row+1, col-1]+c1[row-1, col+1])

    VX= max(c2[row, col]-c2[row+1, col], c2[row, col]-c2[row-1, col], c2[row, col]-c2[row, col+1],
            c2[row, col]-c2[row, col-1],c2[row, col]-c2[row+1, col+1], c2[row, col]-c2[row-1, col+1],
            c2[row, col]-c2[row+1, col-1], c2[row, col]-c2[row-1, col-1])

    VM= min(c4[row, col]-c4[row+1, col], c4[row, col]-c4[row-1, col], c4[row, col]-c4[row, col+1],
            c4[row, col]-c4[row, col-1],c4[row, col]-c4[row+1, col+1], c4[row, col]-c4[row-1, col+1],
            c4[row, col]-c4[row+1, col-1], c4[row, col]-c4[row-1, col-1])

    return VI, VC, VX, VM


if __name__ =='__main__':
    # prepare()
    pixelPrecipData(4)
    # test()
        

