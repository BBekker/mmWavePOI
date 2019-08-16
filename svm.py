from sklearn import svm
import h5py
import sys
import numpy as np


def get_featurevector(data):
    points = np.sum((np.sum(data, axis=2) != 0), axis=1)
    summed = np.sum(data, axis=1)
    averaged = summed / np.sum(data != 0, axis=1)

    featurevecs = np.empty((points.size, 6))
    featurevecs[:,0] = points

    featurevecs[:,1] = averaged[:,0]
    featurevecs[:,2] = averaged[:,1]
    featurevecs[:,3] = averaged[:,2]
    featurevecs[:,4] = averaged[:,3]
    featurevecs[:,5] = summed[:,3]
    #Out: [num points, range, angle, doppler, snr tot, snr avg ]
    return featurevecs


numa = 850
numb = 850

def main():
    datafile = h5py.File(sys.argv[1],'r')
    samples = np.concatenate((datafile['pointclouds'][:numa], datafile['pointcloudsfiets'][:numb]))
    featurevecs = get_featurevector(samples)
    labels = np.array(([0] * numa) + ([1] * numb))
    indices = np.arange(len(labels))
    np.random.default_rng().shuffle(indices)

    print(f"datasets: {numa} {numb}")
    #set up SVM

    model = svm.SVC(kernel='rbf', gamma='auto', C = 0.9, class_weight='balanced')

    x = featurevecs[indices[:-50],:]
    y = labels[indices[:-50]]
    model.fit(x, y)

    trainout = model.predict(x)
    testout = model.predict(featurevecs[indices[-50:],:])

    print(f"train: {np.mean(trainout == y)}")

    # for i in range(100):
    #     print(f"exp: {y[i]} pred: {trainout[i]}")

    print(f"test: {np.mean(testout == labels[indices[-50:]])}")
    # for i in range(100):
    #     print(f"exp: {labels[indices[i+ -100]]} pred: {testout[i]}")


if __name__ == "__main__":
    main()