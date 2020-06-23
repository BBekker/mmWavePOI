import msgpack
import msgpack_numpy
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.feature_selection import f_classif
from lib.confusionMatrix import *
from sys import argv
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import neural_network as nn
from sklearn import manifold
from sklearn import discriminant_analysis
import seaborn as sns

#enable numpy in msgpack files
msgpack_numpy.patch()

files = [#"adult_EWI-25.msgpack",
        #"achtertuinheuvel-1.msgpack" ,
         #"EWI_2_avond-25.msgpack",
       #  "EWI_3-26.msgpack" ,
         #"EWI_solarpanel-29.msgpack",
        #"schoolpleinheuvel-1.msgpack",
    #("ewitest-18.msgpack",1.8,-3.0),
    ("test31-1/mixed-31.msgpack",1.80,-3.3),
    ("test31-1/football_children.msgpack",1.80,-3.3),
    ("test31-1/football_2-31.msgpack",1.80,-3.3),
    ("test31-1/football_3-31.msgpack",1.80,-3.3),
    ("test31-1/adults-31.msgpack",1.80,-3.3),
    ("fietsen-20.msgpack",2.0,-2.6),
    ("fietsen2-20.msgpack",2.0,-2.6),
]

val_files = [
    ("test31-1/one_at_a_time-31.msgpack",1.80,-3.3),
    ("fietsen4-20.msgpack",2.0,-2.6),
]

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def get_featurevector(data, height, angle):
    """
     Data = [range, angle, doppler, snr]
    """
    #print(data)
    #points = np.sum((np.sum(data, axis=2) != 0), axis=1)
    points = data.shape[0]

    summed = np.sum(data, axis=0)
    averaged = np.mean(data,axis=0)
    deviation = np.std(data, axis=0)
    variance = np.var(data, axis=0)

    featurevecs = np.zeros((12))

    x, y = pol2cart(data[:,0], data[:,1])

    _, elevation = pol2cart(data[:,0], data[:,4]+(3.1415/180*angle))
    elevation += height
    featurevecs[0] = averaged[0] #range
    featurevecs[1] = averaged[1] #angle
    featurevecs[2] = averaged[2] #doppler
    featurevecs[3] = np.mean(elevation) #height
    #featurevecs[4] = averaged[3] #snr

    featurevecs[4] = np.std(x)#deviation[0]
    featurevecs[5] = np.std(y)#deviation[1]
    #featurevecs[7] = deviation[2]
    #featurevecs[8] = deviation[3]

    featurevecs[6:9] = variance[0:3]

    featurevecs[10] = np.percentile(elevation, 95)
    featurevecs[11] = np.percentile(elevation, 5)
    #featurevecs[15] = np.mean(data[:,3]) / ((1/(averaged[0]/1400) +130)) if averaged[0] > 6 else (np.mean(data[:,3])/360)
    #featurevecs[7] = averaged[3]
    #featurevecs[16] = summed[3]
    #featurevecs[17] = points
    #Out: [num points, range, angle, doppler, snr tot, snr avg, angle stdev, doppler stdev, rangedev, snr stdev ]

    return featurevecs

featurevector_length= 12

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


numpointsinclouds = []
sequence_length=100

def val_get_dataset(fileinfo, id,group_pointclouds):
    feature_vectors = []
    labels = []
    ids = []
    filename = fileinfo[0]
    sensorheight = fileinfo[1]
    sensorangle = fileinfo[2]
    for msg in read_file(filename):
        msg_feature_vectors = []
        msg_labels = 0
        pointclouds = get_pointclouds(msg)
        if(len(pointclouds)  > 100):
            class_id = msg['class_id']
            if(class_id >=0):
                i = 0
                sequence =[]
                while i < len(pointclouds):
                    pointcloud = pointclouds[i]
                    i += 1
                    if (pointcloud.shape[0] > 1):
                        for j in range(min(group_pointclouds - 1, len(pointclouds)-i)):
                            pointcloud2 = pointclouds[i]
                            if (pointcloud2.shape[0] > 1):
                                pointcloud = np.append(pointcloud, pointcloud2, axis=0)
                            i += 1
                        fv = get_featurevector(pointcloud, sensorheight, sensorangle)
                        sequence.append(fv)

                if(len(sequence) >= (sequence_length//group_pointclouds)):
                    for j in range(0,len(sequence)-sequence_length//group_pointclouds-1,50//group_pointclouds):
                        feature_vectors.append(np.array(sequence[j : j+sequence_length//group_pointclouds]))
                        labels.append(msg['class_id'] if msg['class_id'] >= 0 else 3)
                        ids.append(id*100000+msg['uid'])

    # labels: [adult, bike, child, clutter]

    labels = np.array(labels)
    features = np.array(feature_vectors)
    ids = np.array(ids)
    return labels, features, ids


def get_dataset(fileinfo, id, group_pointclouds):
    feature_vectors = []
    labels = []
    ids = []
    filename = fileinfo[0]
    sensorheight = fileinfo[1]
    sensorangle = fileinfo[2]
    for msg in read_file(filename):
        msg_feature_vectors = []
        msg_labels = 0
        pointclouds = get_pointclouds(msg)
        if(len(pointclouds)  > 50):
            class_id = msg['class_id']
            i = 0
            while i < len(pointclouds):
                pointcloud = pointclouds[i]
                i += 1
                if (pointcloud.shape[0] > 1):
                    for j in range(min(group_pointclouds - 1, len(pointclouds)-i)):
                        pointcloud2 = pointclouds[i]
                        if (pointcloud2.shape[0] > 1):
                            pointcloud = np.append(pointcloud, pointcloud2, axis=0)
                        i += 1
                    numpointsinclouds.append(pointcloud.shape[0])
                    fv = get_featurevector(pointcloud, sensorheight, sensorangle)
                    feature_vectors.append(fv)
                    labels.append(msg['class_id'] if msg['class_id'] >= 0 else 3)
                    ids.append(id*100000+msg['uid'])

    # labels: [adult, bike, child, clutter]

    labels = np.array(labels)
    features = np.array(feature_vectors)
    ids = np.array(ids)
    return labels, features, ids


def showDifference(features, labels, a):
    dataframe = pd.DataFrame(np.hstack((np.expand_dims(labels, axis=1), features)),
                             columns= ["label", "range", "angle", "doppler", "stdx", "stdy", "dopplerstd", "height95", "height5","SNR"])

    dataframe = dataframe[dataframe['label']<3.1]
    labelnames = ["adult","bicycle", "child"]
    dataframe['label'] = dataframe['label'].apply(lambda x:labelnames[int(x)])
    dataframe = dataframe[abs(dataframe['angle'])<1]
    dataframe = dataframe[abs(dataframe['doppler']) > 1]

    sns.set()
    sns.scatterplot("range", "SNR", data=dataframe,hue='label',size=1)
    plt.title("SNR vs Range for 3 classes")
    #
    # sns.jointplot("range", "SNR", data=dataframe[dataframe['label']==2],kind="kde")
    # sns.jointplot("range", "SNR", data=dataframe[dataframe['label']==1],kind="kde")
    # sns.jointplot("range", "SNR", data=dataframe[dataframe['label']==0],kind="kde")
    plt.show()

def getCompleteDataset(files, group_pointclouds):
    features = []
    labels = []
    ids = []
    for j in range(0, len(files), 1):
        a, b, c = get_dataset(files[j], j, group_pointclouds)
        features.append(b)
        labels.append(a)
        ids.append(c)
        #print(b.shape)

    # print(features)
    labels = np.concatenate(labels, axis=0)
    features = np.concatenate(features, axis=0)
    ids = np.concatenate(ids, axis=0)

    #filter out other class
    noother = labels != 3
    labels = labels[noother]
    features = features[noother]
    ids = ids[noother]

    return (features, labels, ids)


def doTheThing(group_pointclouds):
    features, labels, ids = getCompleteDataset(files,group_pointclouds)

    def distributionOfNumPoints():
        sns.set()
        sns.distplot(numpointsinclouds)
        plt.xlim(0,150)
        plt.xlabel("amount of points")
        plt.ylabel("ratio of samples")
        plt.title("Distribution of amount of points in a cluster")
        plt.show()


    print(labels.shape, features.shape)
    np.random.seed(5)
    shuffler = np.arange(labels.shape[0])
    np.random.shuffle(shuffler)

    labels = labels[shuffler]
    features = features[shuffler]
    ids = ids[shuffler]
    #
    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)

    #showDifference(features, labels,0)
    #exit()
    #
    # showDifference(b, a, 1)
    # exit()

    ##TSNE clustering
    # tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    # colors = ['r', 'b', 'g','y']
    # Y = tsne.fit_transform(features)
    # plt.scatter(Y[:, 0], Y[:, 1],color= [colors[x] for x in labels], linewidths=0.01)
    # plt.show()
    # exit()
    # selection = b[:,0] < 20.0
    # a = a[selection]
    # b = b[selection]


    train_labels = labels
    train_features = features

    val_features, val_labels, val_ids = getCompleteDataset(val_files,group_pointclouds)

    val_features = scaler.transform(val_features)
    unique, counts = np.unique(val_labels, return_counts=True)
    print(dict(zip(unique, counts)))

    # ch = chi2(np.abs(train_features), train_labels)
    # print(ch)
    #
    # print(f_classif(train_features, train_labels))
    #select best features
    # feature_selection = RandomForestClassifier(max_depth=20, criterion="entropy", random_state=0,n_estimators=100)
    # feature_selection.fit(train_features, train_labels)
    # print(feature_selection.feature_importances_ )
    # # train_features = train_features[:,feature_selection.get_support()]
    # # val_features = val_features[:,feature_selection.get_support()]
    # exit()
    # print(train_features.shape, val_features.shape)
    #clf = discriminant_analysis.LinearDiscriminantAnalysis(balanced=True)
    clf = RandomForestClassifier(min_impurity_decrease=0.01, criterion="entropy", random_state=0,n_estimators=50)
    #clf = nn.MLPClassifier((128,128), max_iter=500, alpha=0.1)
    #clf = svm.SVC(kernel='rbf', gamma='auto',probability=True)
    #clf = load('randomForrest.joblib')
    clf.fit(train_features,train_labels)


    #Get val set
    val_features = []
    val_labels = []
    for j in range(0, len(val_files), 1):
        a, b, c = val_get_dataset(val_files[j], j,group_pointclouds)
        val_features.append(b)
        val_labels.append(a)
        print(b.shape)

    val_labels = np.concatenate(val_labels, axis=0)
    val_features = np.concatenate(val_features, axis=0)

    print(val_features.shape)

    results = []
    for i in range(val_labels.shape[0]):
        feat = val_features[i]
        #print(feat.shape)
        probs = clf.predict_proba(feat)
        avg = np.mean(probs,axis=0)
        print(avg)
        print(val_labels[i])
        #results.append(val_labels[i] == np.argmax(avg))

    print(np.mean(results))


for i in range(10,20):
    print("===============")
    print(i)
    doTheThing(i)
