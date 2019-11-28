from sklearn import svm
from sklearn import neural_network as nn
from sklearn.preprocessing import StandardScaler
import h5py
import sys
import numpy as np
from matplotlib import pyplot as plt
from joblib import dump, load
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def get_featurevector(data):
    """
     Data = [range, angle, doppler, snr]
    """
    data = data[:,:,:4]
    #print(data)
    points = np.sum((np.sum(data, axis=2) != 0), axis=1)

    summed = np.sum(data, axis=1)
    averaged = summed / np.tile(points, [4,1]).T
    #deviation = np.std(data, axis=1)

    featurevecs = np.zeros((data.shape[0], 12))

    featurevecs[:,0] = points
    featurevecs[:,1] = averaged[:,0]
    featurevecs[:,2] = averaged[:,1]
    featurevecs[:,3] = averaged[:,2]
    featurevecs[:,4] = averaged[:,3]
    featurevecs[:,5] = summed[:,3]

    for i in range(data.shape[0]):
        featurevecs[i,6] = np.std(data[i,:points[i],1])
        featurevecs[i,7] = np.std(data[i,:points[i],2])
        featurevecs[i,8] = np.std(data[i,:points[i],0])
        featurevecs[i,9] = np.std(data[i,:points[i],3])
    #Out: [num points, range, angle, doppler, snr tot, snr avg, angle stdev, doppler stdev, rangedev, rssi stdev ]


    return featurevecs


def get_featurevector2(data):
    points = np.sum((np.sum(data, axis=2) != 0), axis=1)
    
    featurevecs = np.zeros((data.shape[0], 3))

    for i in range(data.shape[0]):
        featurevecs[i,0] = np.mean(data[i,:points[i], 3] * data[i,:points[i],0]**2)
        featurevecs[i,1] = np.max(data[i,:points[i], 3] * data[i,:points[i],0]**2)
        featurevecs[i,2] = np.sum(data[i,:points[i], 3] * data[i,:points[i],0]**2)
    return featurevecs

def get_featurevector3(data):
    """
     Data = [range, angle, doppler, snr]
    """
    points = np.sum((np.sum(data, axis=2) != 0), axis=1)

    summed = np.sum(data, axis=1)
    averaged = summed / np.tile(points, [4,1]).T
    #deviation = np.std(data, axis=1)
 
    featurevecs = np.zeros((data.shape[0], 11))

    featurevecs[:,0] = points
    featurevecs[:,1] = averaged[:,0]
    featurevecs[:,2] = averaged[:,1]
    featurevecs[:,3] = averaged[:,2]
    featurevecs[:,4] = averaged[:,3]
    featurevecs[:,5] = summed[:,3]
    featurevecs[:,10] = np.max(data[:,:,3], axis=1)

    for i in range(data.shape[0]):
        featurevecs[i,6] = np.std(data[i,:points[i],1])
        featurevecs[i,7] = np.std(data[i,:points[i],2])
        featurevecs[i,8] = np.std(data[i,:points[i],0])
        featurevecs[i,9] = np.std(data[i,:points[i],3])

        # featurevecs[i,10] = np.min(data[i,:points[i],2])
        # featurevecs[i,11] = np.max(data[i,:points[i],2])
    #Out: [num points, range, angle, doppler, snr tot, snr avg, angle stdev, doppler stdev, rangedev, rssi stdev ]


    return featurevecs


