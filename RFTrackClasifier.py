import msgpack
import msgpack_numpy
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import pandas as pd

from lib.confusionMatrix import *
from sys import argv
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import neural_network as nn
from sklearn import manifold

#enable numpy in msgpack files
msgpack_numpy.patch()

files = [#"adult_EWI-25.msgpack",
        #"achtertuinheuvel-1.msgpack" ,
         #"EWI_2_avond-25.msgpack",
       #  "EWI_3-26.msgpack" ,
         #"EWI_solarpanel-29.msgpack",
        #"schoolpleinheuvel-1.msgpack",
         "ewitest-18.msgpack"
]

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

    featurevecs = np.zeros((7))

    featurevecs[0] = averaged[0]
    featurevecs[1] = averaged[1]
    featurevecs[2] = averaged[2]
    featurevecs[3] = averaged[3]
    featurevecs[4] = summed[3] / 10
    featurevecs[5] = np.percentile(data[:,4], 90)
    featurevecs[6] = np.mean(data[:,3]) / ((1/(averaged[0]/1300) +130))
    #Out: [num points, range, angle, doppler, snr tot, snr avg, angle stdev, doppler stdev, rangedev, snr stdev ]

    return featurevecs

featurevector_length= 7

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

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

group_pointclouds = 5

def get_dataset(filename):
    labels = []
    feature_vectors = []

    for msg in read_file(filename):
        msg_feature_vectors = []
        msg_labels = 0
        pointclouds = get_pointclouds(msg)
        listified = [x for x in pointclouds if (x.shape[0] > 1)]
        if(len(listified) >20):
            supercloud = np.concatenate(listified)
            if(msg['class_id'] >= 0):
                #print(np.percentile(supercloud[:, 4], 90), msg['class_id'])
                labels.append(msg['class_id'])
                _, elevation = pol2cart(supercloud[:, 0], supercloud[:, 4])
                #feature_vectors.append(np.percentile(supercloud[:,3] / ((1/(supercloud[:,0]/1300) +130)),95))
                feature_vectors.append
            # for pointcloud in pointclouds:
            #     percentiles
            #     fv = get_featurevector(pointcloud)
            #     msg_feature_vectors.append(fv)
            #     labels.append(msg['class_id'] if msg['class_id'] >= 0 else 3)


    # labels: [adult, bike, child, clutter]

    a = np.array(labels)
    b = np.array(feature_vectors)
    return a, b

def showDifference(features, labels, sel):
    dataframe = pd.DataFrame(np.hstack((np.expand_dims(labels, axis=1), features)),
                             columns= ["label", "range", "angle", "doppler", "snravg", "snrsum", "height", "RCS" ])
    grouped = dataframe.groupby('label')['range','RCS']
    print(grouped)
    grouped.boxplot()
    plt.show()

features = []
labels = []
for j in range(0, len(files), 1):
    a, b = get_dataset("labeling/"+files[j])
    features.append(b)
    labels.append(a)
    print(b.shape)
#print(features)
a = np.concatenate(labels, axis=0)
b = np.concatenate(features,axis=0)

plt.plot(b[a==0])
plt.plot(b[a==2])
plt.show()

exit()
print(a.shape, b.shape)
np.random.seed(1)
shuffler = np.arange(a.shape[0])
np.random.shuffle(shuffler)

a = a[shuffler]
b = b[shuffler]

val_samples = a.shape[0] // 5

#
# showDifference(b, a, 1)
# exit()
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
# selection = b[:,1] < 25.0
# a = a[selection]
# b = b[selection]
# #
# colors = ['r', 'b', 'g','y']
# Y = tsne.fit_transform(b)
# plt.scatter(Y[:, 0], Y[:, 1],color= [colors[x] for x in a])
# plt.show()


selection = b[:,1] < 20.0
a = a[selection]
b = b[selection]

noother = a != 3
a = a[noother]
b = b[noother]

train_labels = a[val_samples:]
train_features = b[val_samples:]



val_labels = a[:val_samples]
val_features = b[:val_samples]

unique, counts = np.unique(val_labels, return_counts=True)
print(dict(zip(unique, counts)))
#select best features
feature_selection = SelectFromModel(ExtraTreesClassifier(n_estimators=50),max_features=7)
feature_selection.fit(train_features, train_labels)

train_features = train_features[:,feature_selection.get_support()]
val_features = val_features[:,feature_selection.get_support()]

print(train_features.shape, val_features.shape)
clf = RandomForestClassifier(max_depth=20, criterion="entropy", random_state=0,n_estimators=100)
#clf = nn.MLPClassifier((100,100,100), max_iter=300, alpha=0.10)
#clf = svm.SVC(kernel='rbf', gamma='auto', C = 0.95, class_weight='balanced', probability=True)
#clf = load('randomForrest.joblib')
clf.fit(train_features,train_labels)
#dump(clf, 'randomForrest.joblib')
#print(a.shape, np.argmax(res.numpy(),axis=1).shape)

print("feature importance: ", clf.feature_importances_)
print("Train:")
plot_confusion_matrix(train_labels, clf.predict(train_features), [ 'adult', 'bicycle','child', 'unlabled'])

print("val:")
validation_pred = clf.predict(val_features)
plot_confusion_matrix(val_labels, validation_pred, ['adult', 'bicycle', 'child','unlabeled'], normalize=False)
plt.show()
