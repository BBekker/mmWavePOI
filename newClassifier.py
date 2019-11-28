import msgpack
import msgpack_numpy
import numpy as np

#enable numpy in msgpack files
msgpack_numpy.patch()

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

    featurevecs = np.zeros((10))

    featurevecs[0] = points
    featurevecs[1] = averaged[0]
    featurevecs[2] = averaged[1]
    featurevecs[3] = averaged[2]
    featurevecs[4] = averaged[3]
    featurevecs[5] = summed[3]

    featurevecs[6] = np.std(data[:,1])
    featurevecs[7] = np.std(data[:,2])
    featurevecs[8] = np.std(data[:,0])
    featurevecs[9] = np.std(data[:,3])
    #Out: [num points, range, angle, doppler, snr tot, snr avg, angle stdev, doppler stdev, rangedev, rssi stdev ]

    return featurevecs

featurevector_length= 10

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

for msg in read_file("ewi-8.msgpack"):
    pointclouds = get_pointclouds(msg)

    feature_vectors = np.zeros((len(pointclouds), featurevector_length))
    for i in range(len(pointclouds)):
        pointcloud = pointclouds[i]
        if(pointcloud.shape[0] > 1):
            feature_vectors[i] = get_featurevector(pointcloud)
    print(feature_vectors)