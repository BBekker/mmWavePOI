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

def countPointclouds(pcs):
    count = 0
    for pc in pcs:
        if(pc.size > 0):
            count += 1
    return count

if __name__ == "__main__":
    msgpack_numpy.patch()
    xys = []
    all_pointclouds = []

    angle=0.0

    lengths = []
    with open(sys.argv[1], 'rb') as file:
        unpacker = msgpack.Unpacker(file, raw=False)

        data = {}

        for msg in unpacker:
            if(msg['class_id'] > -1):
                pointclouds = msg['pointclouds']


                if(countPointclouds(pointclouds) > 200):
                    elevations = []
                    distances = []
                    for data in pointclouds:
                        if data.size > 0:
                            distance, elevation = pol2cart(data[:, 0], data[:, 4] + (3.1415 / 180 * angle))
                            elevations += [np.percentile(elevation,95)]
                            distances += [np.mean(distance)]

                    #filteredclouds = [np.mean(x[:,3]) for x in pointclouds if x.size>1]
                    plt.plot(distances,elevations)
                    plt.show()


    # plt.scatter(averages[:,0], maxes[:,3] / (1/(averages[:,0]/1300) +130))
    #
    # x = np.linspace(5,50,100)
    # #plt.plot(x,1/(x/1300) +130)
    # plt.show()


