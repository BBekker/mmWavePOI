from mpl_toolkits.mplot3d import Axes3D
import h5py
import sys
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

def get_featurevector(data):
    points = np.sum((np.sum(data, axis=2) != 0), axis=1)

    summed = np.sum(data, axis=1)
    averaged = summed / np.tile(points, [4,1]).T
    maxed = np.max(data, axis=1)
    #deviation = np.std(data, axis=1)

    featurevecs = np.zeros((data.shape[0], 11))

    featurevecs[:,0] = points
    featurevecs[:,1] = averaged[:,0]
    featurevecs[:,2] = averaged[:,1]
    featurevecs[:,3] = averaged[:,2]
    featurevecs[:,4] = averaged[:,3]
    featurevecs[:,5] = summed[:,3]
    #featurevecs[:,6] = deviation[:,1]
    #featurevecs[:,7] = deviation[:,2]
    featurevecs[:,10] = maxed[:, 3]

    for i in range(data.shape[0]):
        #featurevecs[i,3] = np.std(data[i,:points[i],2])
        featurevecs[i,6] = np.std(data[i,:points[i],1])
        featurevecs[i,7] = np.std(data[i,:points[i],2])
        featurevecs[i,8] = np.std(data[i,:points[i],0])
        featurevecs[i,9] = np.std(data[i,:points[i],3])

    #Out: [num points, range, angle, doppler, snr tot, snr avg, angle stdev, doppler stdev, rangedev, rssi stdev ]


    return featurevecs


def main():
    datafile1 = h5py.File(sys.argv[1],'r')
    datafile2 = h5py.File(sys.argv[2], 'r')

    pc1 = datafile1['pointclouds/samples'][500:4500]
    pc2 = datafile2['pointclouds/samples'][0:4000]

    numa = pc1.shape[0]  
    numb = pc2.shape[0]
    
    samples = np.concatenate((pc2, pc1))
    
    X = get_featurevector(samples)

    
    #print(featurevecs)
    labels = np.array(([1] * numa) + ([0] * numb))
    

    #Normalize our input values
    # scaler = StandardScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)

    #highpass
    b, a = signal.butter(2, 0.02, btype='high')

    # fig = plt.figure()
    # # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(xs= X[:,1], ys = X[:,2], zs=X[:,10], c = labels)

    doppler =X[:,3]

    plt.plot(range(len(X)), doppler, range(len(X)), X[:,1] )
    plt.show()

    f, t, Sxx = signal.spectrogram(doppler, 10.0, nperseg=128, noverlap=64)
    plt.pcolormesh(t, f, Sxx,vmin=0, vmax=1)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()



if __name__ == "__main__":
    main()  