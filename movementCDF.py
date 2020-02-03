import msgpack
import msgpack_numpy
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import pandas as pd

from lib.confusionMatrix import *
from sys import argv
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import neural_network as nn
from sklearn import manifold

#enable numpy in msgpack files
msgpack_numpy.patch()

files = [#"adult_EWI-25.msgpack",
        #"achtertuinheuvel-1.msgpack" ,
        #"EWI_2_avond-25.msgpack",
       #  "EWI_3-26.msgpack" ,
         #"EWI_solarpanel-29.msgpack",
        #"schoolpleinheuvel-1.msgpack",
         "ewitest-18.msgpack"
]

def get_featurevector(data):
    """
     Data = [range, angle, doppler, snr]
    """
    #print(data)
    #points = np.sum((np.sum(data, axis=2) != 0), axis=1)
    points = data.shape[0]

    summed = np.sum(data, axis=0)
    averaged = summed / points
    #deviation = np.std(data, axis=1)

    featurevecs = np.zeros((7))

    featurevecs[0] = averaged[0]
    featurevecs[1] = averaged[1]
    featurevecs[2] = np.abs(averaged[2])
    featurevecs[3] = averaged[3]
    featurevecs[4] = summed[3] / 10
    featurevecs[5] = np.percentile(data[:,4], 90)
    featurevecs[6] = np.mean(data[:,3]) / ((1/(averaged[0]/1300) +130))
    #Out: [num points, range, angle, doppler, snr tot, snr avg, angle stdev, doppler stdev, rangedev, snr stdev ]

    return featurevecs

featurevector_length= 7

def read_file(filename):
    """
    read a messagepack file and return individual messages
    :return: 
    """
    with open(filename, 'rb') as file:
        unpacker = msgpack.Unpacker(file, raw=False)
        for msg in unpacker:
            yield msg


def get_pointclouds(msg):
    """
    get pointcloud data from msg
    :param msg:
    :return:
    """
    return msg['pointclouds']

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

group_pointclouds = 1

def get_dataset(filename):
    labels = []
    feature_vectors = []

    for msg in read_file(filename):
        msg_feature_vectors = []
        msg_labels = 0
        pointclouds = get_pointclouds(msg)
        doppler_values = []

        if(len(pointclouds) > 300):
            class_id = msg['class_id']
            i = 0
            while i < len(pointclouds):
                pointcloud = pointclouds[i]
                i += 1
                if (pointcloud.shape[0] > 1):
                    #get movement
                    doppler_values.append(np.abs(np.mean(pointcloud[:,1])))

            sorted_values = np.sort(doppler_values)
            sorted_counts = np.arange(len(sorted_values)) / len(sorted_values)
            integrated = np.trapz(sorted_counts, sorted_values) + (2.0 - sorted_values[-1])
            feature_vectors.append(integrated)
            labels.append(msg['class_id'] if msg['class_id'] >= 0 else 3)



    # labels: [adult, bike, child, clutter]

    a = np.array(labels)
    b = np.array(feature_vectors)
    return a, b

features = []
labels = []
for j in range(0, len(files), 1):
    a, b = get_dataset("labeling/"+files[j])
    features.append(b)
    labels.append(a)
    print(b.shape)
#print(features)
a = np.concatenate(labels, axis=0)
b = np.concatenate(features,axis=0)

plt.boxplot([b[a==0],b[a==1],b[a==2]],labels=['adult','bike','child'])
plt.show()