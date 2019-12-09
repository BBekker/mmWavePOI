import msgpack
import msgpack_numpy
import numpy as np
from sys import argv
import tensorflow as tf


#enable numpy in msgpack files
msgpack_numpy.patch()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])



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


def get_dataset(filename):
    feature_vectors = []
    labels = []

    for msg in read_file(filename):
        pointclouds = get_pointclouds(msg)
        if(len(pointclouds) > 20):
            class_id = msg['class_id']
            for i in range(len(pointclouds)):
                pointcloud = pointclouds[i]
                if (pointcloud.shape[0] > 1):
                    feature_vectors.append(get_featurevector(pointcloud))
                    labels.append(msg['class_id'] if msg['class_id'] >= 0 else 3)

    # labels: [adult, bike, child, clutter]

    a = tf.one_hot(np.array(labels), depth=4)
    b = np.stack(feature_vectors)
    return a, b


features = []
labels = []
for j in range(1, len(argv), 1):
    a, b = get_dataset(argv[j])
    features.append(b)
    labels.append(a)
    print(b.shape)

#print(features)
a = tf.concat(labels, axis=0)
b = np.concatenate(features,axis=0)

b = tf.keras.utils.normalize(b, axis=0, order=2)
print(a.shape, b.shape)
dataset = tf.data.Dataset.from_tensor_slices((b,a))


BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)
test_dataset = dataset.take(5000).batch(BATCH_SIZE)
train_dataset = dataset.skip(5000).batch(BATCH_SIZE)

# print(train_dataset)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002,
                                                    beta_1=0.9,
                                                    beta_2=0.999,
                                                    epsilon=1e-07,),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalAccuracy()])

#model.load_weights('easy_checkpoint')
model.fit(train_dataset, epochs=450, validation_data=test_dataset)
model.save_weights('easy_checkpoint')
res = model(b)
#print(a.shape, np.argmax(res.numpy(),axis=1).shape)
print(tf.math.confusion_matrix(np.argmax(a,axis=1), np.argmax(res.numpy(),axis=1)))