from sklearn import svm
from sklearn import neural_network as nn
from sklearn.preprocessing import StandardScaler
import h5py
import sys
import numpy as np
from matplotlib import pyplot as plt
from joblib import dump, load
from sklearn.metrics import f1_score
from lib.classifier import *
from lib.confusionMatrix import *

addRandom = False

#Note 3 is the random class
datasets_config = [   #Filename               Class
    ("data/200819dataset.hdf5",       0),
    ("data/adult20190909154014.hdf5", 1),
    ("data/adult200919.hdf5",         1),
    ("data/fiets20190909212034.hdf5", 2),
    #("adultvalidation20190909213834.hdf5", 1),
]

def main():

    datafiles = [h5py.File(x[0],'r') for x in datasets_config]
    pointclouds = [x['pointclouds/samples'][:] for x in datafiles]

    samples = np.concatenate(tuple(pointclouds))
    labels = np.array([])
    for i in range(len(datasets_config)):
        labels= np.append(labels, np.ones(len(pointclouds[i])) * datasets_config[i][1])
        print(f"Loaded {datasets_config[i][0]} with {len(pointclouds[i])} samples of class {datasets_config[i][1]} ")

    
    featurevecs = get_featurevector(samples)

    #Normalize our input values
    scaler = StandardScaler()
    scaler.fit(featurevecs)
    featurevecs = scaler.transform(featurevecs)
    dump(scaler, "scaler.joblib")

    

    #Add random adversarial examples, same shape as our feature vecs
    if addRandom:
        random = np.random.normal(0.0, 1.0, featurevecs.shape)
        random_labels = np.ones(random.shape[0]) * 3
        
        featurevecs = np.concatenate((featurevecs, random))
        labels = np.concatenate((labels, random_labels))

    #split up data
    indices = np.arange(len(labels))

    np.random.seed(1337)
    np.random.shuffle(indices)

    trainset = featurevecs[indices[:-4000],:]
    trainlabels = labels[indices[:-4000]]

    testset = featurevecs[indices[-4000:],:]
    testlabels = labels[indices[-4000:]]

    #Train
    print("start train")
    #model = svm.SVC(kernel='rbf', gamma='auto', C = 0.95, class_weight='balanced', probability=True)
    model = nn.MLPClassifier((100,100,100), max_iter=300, alpha=0.10)
    x = trainset
    y = trainlabels
    #model.fit(x, y)
    model = load('model.joblib') #Use model from previous run instead
    trainout = model.predict(x)
    testout = model.predict(testset)
    probs = model.predict_proba(testset)


    #Viz and postprocess

    print(f"train acc: {np.mean(trainout == trainlabels)}")

    # for i in range(100):
    #     print(f"exp: {y[i]} pred: {trainout[i]}")

    print(f"test acc: {np.mean(testout == testlabels)}")
    
    print(f"f-score: {f1_score(testlabels, testout, average='macro')}")
    #for i in range(100):
    #     print(f"features: {featurevecs[indices[i]]} exp: {labels[indices[i]]} out: {testout[indices[i]]} pred: {probs[i]}")

    # plt.scatter(featurevecs[indices[:-500],4], featurevecs[indices[:-500],5], c=trainout)
    # plt.show()


    #graph(featurevecs[indices[:-500],4], featurevecs[indices[:-500],5], featurevecs[indices[:-500],1], trainout)
    #dump(model, 'model.joblib')

    # numcorrect
    # for i in range(len(testout)):


    #MSE
    error = np.sqrt(np.mean(np.power(np.ones(probs.shape[0]) - probs[:,np.argmax(probs, axis=1)], 2)))

    print(error)
    #moving average results
    
    plot_confusion_matrix(testlabels, testout, ['child','adult','bicycle'])
    plt.show()
    sortedtestindices = np.sort(indices[-4000:])
    probs2 = model.predict_proba(featurevecs[sortedtestindices])
    
    n_avg = 10
    avgprobs = np.apply_along_axis(lambda m: np.convolve(m, np.ones((n_avg,))/n_avg, mode='same'), axis=0, arr=probs2)

    result = np.argmax(avgprobs, axis=1)
    correct = result == labels[sortedtestindices]
    print("N", n_avg, "acc", np.sum(correct) / len(result), "f1:", f1_score(labels[sortedtestindices], result, average='macro'))
    # plt.plot(range(probs2.shape[0]), avgprobs[:,1], range(probs2.shape[0]), labels[sortedtestindices])
    # plt.show()


if __name__ == "__main__":
    main()