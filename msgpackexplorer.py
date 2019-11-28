import msgpack
import msgpack_numpy
import numpy as np
import sys
from matplotlib import pyplot as plt


if __name__ == "__main__":
    msgpack_numpy.patch()
    xys = []
    tracks = 0
    longtracks = 0
    with open(sys.argv[1], 'rb') as file:
        unpacker = msgpack.Unpacker(file, raw=False)
        for msg in unpacker:
            tracks +=1
            if(msg['class_id'] > -1):
                longtracks += 1

                xy = np.array(msg['track'])
                plt.scatter(xy[:,0], xy[:,1], edgecolors='face')
    #plt.scatter(np.array(xys))
    print(tracks, longtracks)
    plt.show()