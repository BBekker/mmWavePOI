from sklearn import svm
from sklearn.preprocessing import StandardScaler
import h5py
import sys
import numpy as np


def get_featurevector(data):
    points = np.sum((np.sum(data, axis=2) != 0), axis=1)
    summed = np.sum(data, axis=1)
    averaged = summed / np.sum(data != 0, axis=1)
    deviation = np.std(data, axis=1)

    featurevecs = np.empty((points.size, 6))

    featurevecs[:,0] = points
    featurevecs[:,1] = averaged[:,0]
    featurevecs[:,2] = averaged[:,1]
    featurevecs[:,3] = deviation[:,2]
    featurevecs[:,4] = np.log(averaged[:,3])
    featurevecs[:,5] = np.log(summed[:,3])
    #Out: [num points, range, angle, doppler, snr tot, snr avg ]
    return featurevecs


numa = 650
numb = 650

def main():
    datafile = h5py.File(sys.argv[1],'r')
    samples = np.concatenate((datafile['pointclouds'][:numa], datafile['pointcloudsfiets'][:numb]))
    featurevecs = get_featurevector(samples)
    labels = np.array(([0] * numa) + ([1] * numb))
    indices = np.arange(len(labels))
    np.random.shuffle(indices)

    print(f"datasets: {numa} {numb}")
    #set up SVM

    scaler = StandardScaler()
    scaler.fit(featurevecs)
    print(scaler.get_params())
    featurevecs = scaler.transform(featurevecs)

    model = svm.SVC(kernel='rbf', gamma='auto', C = 0.95, class_weight='balanced', probability=True)

    x = featurevecs[indices[:-150],:]
    y = labels[indices[:-150]]
    model.fit(x, y)

    trainout = model.predict(x)
    testout = model.predict(featurevecs[indices[-150:],:])
    probs = model.predict_proba(featurevecs[indices[-150:],:])

    print(f"train: {np.mean(trainout == y)}")

    # for i in range(100):
    #     print(f"exp: {y[i]} pred: {trainout[i]}")

    print(f"test: {np.mean(testout == labels[indices[-150:]])}")
    # for i in range(100):
    #     print(f"exp: {labels[indices[i+ -100]]} pred: {testout[i]}")


if __name__ == "__main__":
    main()