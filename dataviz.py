from mpl_toolkits.mplot3d import Axes3D
import h5py
import sys
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from lib.classifier import *
import matplotlib.colors as mcolors
from scipy import optimize


def test_func(x, a, b):
    return a / (x)**b

def main():
    datafile1 = h5py.File(sys.argv[1],'r')
    datafile2 = h5py.File(sys.argv[2], 'r')

    pc1 = datafile1['pointclouds/samples'][500:7500]
    pc2 = datafile2['pointclouds/samples'][0:7000]

    numa = pc1.shape[0]  
    numb = pc2.shape[0]
    
    X = []
    X.append(get_featurevector(pc1))
    X.append(get_featurevector(pc2))

    
    #print(featurevecs)
    labels = np.array(([0] * numa) + ([1] * numb))
    

    #Normalize our input values
    # scaler = StandardScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)

    colors = ["tab:red", "tab:blue"]

    fig, ax = plt.subplots()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(xs= X[:,1], ys = np.abs(X[:,2]), zs=X[:,12], c = labels)
    plotlabels=  np.array(['children','adults'])
    for i in range(0, len(sys.argv)-1):
        print(i)
        vfilter = np.abs(X[i][:,2]) < 0.08
        ax.scatter(X[i][vfilter,1], X[i][vfilter,4], c=[colors[i]]*np.sum(vfilter), label=plotlabels[i], alpha=0.6, marker='+')
        
        params, params_covariance = optimize.curve_fit(test_func, X[i][vfilter,1], X[i][vfilter,4], p0=[30, 2])
        ax.plot(np.arange(2,30,0.1), test_func(np.arange(2,30,0.1), params[0], params[1]),color=colors[i],linewidth=4.0)
        ax.set_xlabel("distance [m]")
        ax.set_ylabel("μ RSSI")
    plt.legend()
    plt.show()
    # doppler =X[:,3]

    # plt.plot(range(len(X)), X[:,1], range(len(X)), X[:,2], range(len(X)), X[:,12] )
    # plt.show()

    # f, t, Sxx = signal.spectrogram(doppler, 10.0, nperseg=128, noverlap=64)
    # plt.pcolormesh(t, f, Sxx,vmin=0, vmax=1)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()



if __name__ == "__main__":
    main()  