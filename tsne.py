from sklearn.manifold import TSNE
from sklearn import neural_network as nn
from sklearn.preprocessing import StandardScaler
import h5py
import sys
import numpy as np
from matplotlib import pyplot as plt
from joblib import dump, load

def get_featurevector(data):
    points = np.sum((np.sum(data, axis=2) != 0), axis=1)

    summed = np.sum(data, axis=1)
    averaged = summed / np.tile(points, [4,1]).T
    #deviation = np.std(data, axis=1)

    featurevecs = np.zeros((data.shape[0], 8))

    featurevecs[:,0] = points
    featurevecs[:,1] = averaged[:,0]
    featurevecs[:,2] = averaged[:,1]
    #featurevecs[:,3] = deviation[:,2]
    featurevecs[:,4] = averaged[:,3]
    featurevecs[:,5] = summed[:,3]
    #featurevecs[:,6] = deviation[:,1]
    #featurevecs[:,7] = deviation[:,2]

    for i in range(data.shape[0]):
        featurevecs[i,3] = np.std(data[i,:points[i],2])
        featurevecs[i,6] = np.std(data[i,:points[i],1])
        featurevecs[i,7] = np.std(data[i,:points[i],2])

    #Out: [num points, range, angle, doppler, snr tot, snr avg ]


    return featurevecs


def main():
    datafile1 = h5py.File(sys.argv[1],'r')
    datafile2 = h5py.File(sys.argv[2], 'r')

    pc1 = datafile1['pointclouds/samples'][:3500]
    pc2 = datafile2['pointclouds/samples'][:]

    numa = pc1.shape[0]
    numb = pc2.shape[0]
    
    labels = np.array(([0] * numa) + ([1] * numb))
    samples = np.concatenate((pc1, pc2))
    
    featurevecs = get_featurevector(samples)

    embedded = TSNE(n_components=2).fit_transform(featurevecs)
    plt.scatter(embedded[:,0], embedded[:,1], c=labels)
    plt.show()



if __name__ == "__main__":
    main()