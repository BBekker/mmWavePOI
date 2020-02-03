import msgpack
import msgpack_numpy
import numpy as np
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

if __name__ == "__main__":
    msgpack_numpy.patch()
    xys = []
    all_pointclouds = []
    with open(sys.argv[1], 'rb') as file:
        unpacker = msgpack.Unpacker(file, raw=False)

        data = {}

        for msg in unpacker:
            if(msg['class_id'] > -1):
                pointclouds = msg['pointclouds']
                for pointcloud in pointclouds:
                    if (pointcloud.shape[0] > 1):
                        if msg['class_id'] == 0:
                            all_pointclouds.append(pointcloud)

    averages = np.array([x.mean(axis=0) for x in all_pointclouds])
    maxes = np.array([x.sum(axis=0) for x in all_pointclouds])
    print(averages.shape)
    plt.scatter(averages[:,0], maxes[:,3] / (1/(averages[:,0]/1300) +130))

    x = np.linspace(5,50,100)
    #plt.plot(x,1/(x/1300) +130)
    plt.show()


