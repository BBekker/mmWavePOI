from sklearn import svm
from sklearn import neural_network as nn
from sklearn.preprocessing import StandardScaler
import h5py
import sys
import numpy as np
from matplotlib import pyplot as plt
from joblib import dump, load
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

datasets_config = [   #Filename               Class
    ("200819dataset.hdf5",       0),
    ("adult20190909154014.hdf5", 1),
    ("adult200919.hdf5",         1),
    ("fiets20190909212034.hdf5", 2)
]


def get_featurevector(data):
    points = np.sum((np.sum(data, axis=2) != 0), axis=1)

    summed = np.sum(data, axis=1)
    averaged = summed / np.tile(points, [4,1]).T
    #deviation = np.std(data, axis=1)

    featurevecs = np.zeros((data.shape[0], 10))

    featurevecs[:,0] = points
    featurevecs[:,1] = averaged[:,0]
    featurevecs[:,2] = averaged[:,1]
    featurevecs[:,3] = averaged[:,2]
    featurevecs[:,4] = averaged[:,3]
    featurevecs[:,5] = summed[:,3]

    for i in range(data.shape[0]):
        #featurevecs[i,3] = np.std(data[i,:points[i],2])
        featurevecs[i,6] = np.std(data[i,:points[i],1])
        featurevecs[i,7] = np.std(data[i,:points[i],2])
        featurevecs[i,8] = np.std(data[i,:points[i],0])
        featurevecs[i,9] = np.std(data[i,:points[i],3])

    #Out: [num points, range, angle, doppler, snr tot, snr avg, angle stdev, doppler stdev, rangedev, rssi stdev ]


    return featurevecs

def filtered(dataset):
    index =np.product(np.sum(dataset != 0, axis=1) != 0, axis=1)
    return dataset[index]

def graph(x, y, z, color):
    xyz = np.vstack((x,y,z)).T
    colormap = [[1.0,0.0,0.0,0.5],[1.0,1.0,0.0,.5],[0.0,1.0,0.0,.5],[0.0,1.0,1.0,.5],[1.0,0.0,1.0,.5]]

    import pyqtgraph as pg
    pg.mkQApp()

    ## make a widget for displaying 3D objects
    import pyqtgraph.opengl as gl
    view = gl.GLViewWidget()
    scatter = gl.GLScatterPlotItem(pos=xyz)
    view.addItem(scatter)

    view.show()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

numa = 650
numb = 650

def main():

    datafiles = [h5py.File(x[0],'r') for x in datasets_config]
    pointclouds = [x['pointclouds/samples'][:] for x in datafiles]

    samples = np.concatenate(tuple(pointclouds))
    labels = np.array([])
    for i in range(len(datasets_config)):
        labels= np.append(labels, np.ones(len(pointclouds[i])) * datasets_config[i][1])
        print(f"Loaded {datasets_config[i][0]} with {len(pointclouds[i])} samples of class {datasets_config[i][1]} ")

    
    featurevecs = get_featurevector(samples)


    indices = np.arange(len(labels))

    np.random.seed(1337)
    np.random.shuffle(indices)
    #set up SVM

    #Normalize our input values
    scaler = StandardScaler()
    scaler.fit(featurevecs)
    featurevecs = scaler.transform(featurevecs)

    #split up data
    
    trainset = featurevecs[indices[:-2000],:]
    trainlabels = labels[indices[:-2000]]

    testset = featurevecs[indices[-2000:],:]
    testlabels = labels[indices[-2000:]]

    #Train
    print("start train")
    #model = svm.SVC(kernel='rbf', gamma='auto', C = 0.95, class_weight='balanced', probability=True)
    model = nn.MLPClassifier((100,100), max_iter=250, alpha=0.2)
    x = trainset
    y = trainlabels
    #model.fit(x, y)
    model = load('model.joblib') #Use model from previous run instead

    trainout = model.predict(x)
    testout = model.predict(testset)
    #probs = model.predict_proba(featurevecs[indices,:])


    #Viz and postprocess

    print(f"train: {np.mean(trainout == trainlabels)}")

    # for i in range(100):
    #     print(f"exp: {y[i]} pred: {trainout[i]}")

    print(f"test: {np.mean(testout == testlabels)}")
    #for i in range(100):
    #     print(f"features: {featurevecs[indices[i]]} exp: {labels[indices[i]]} out: {testout[indices[i]]} pred: {probs[i]}")

    # plt.scatter(featurevecs[indices[:-500],4], featurevecs[indices[:-500],5], c=trainout)
    # plt.show()


    #graph(featurevecs[indices[:-500],4], featurevecs[indices[:-500],5], featurevecs[indices[:-500],1], trainout)
    dump(model, 'model.joblib')

    # numcorrect
    # for i in range(len(testout)):

    #moving average results
    
    plot_confusion_matrix(testlabels, testout, ['child','adult','bicycle'])
    plt.show()
    probs2 = model.predict_proba(featurevecs)
    n_avg = 5
    avgprobs = np.apply_along_axis(lambda m: np.convolve(m, np.ones((n_avg,))/n_avg, mode='same'), axis=0, arr=probs2)

    result = np.argmax(avgprobs, axis=1)
    correct = result == labels
    print(result, len(result))
    print(labels.astype(int), len(labels))
    print(result == labels.astype(int))
    print(np.sum(result == labels) / len(result))
    plt.plot(range(probs2.shape[0]), probs2[:,1], range(probs2.shape[0]), labels[:probs2.shape[0]])
    plt.show()


if __name__ == "__main__":
    main()