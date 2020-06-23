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
    #("EWI_3-26.msgpack",2.0,-2) ,
         #"EWI_solarpanel-29.msgpack",
        #"schoolpleinheuvel-1.msgpack",
    #("labeling/ewitest-18.msgpack",2.0,0.0),
    ("test31-1/mixed-31.msgpack",1.8,-3.3),
    ("test31-1/football_children.msgpack",1.80,-3.3),
    ("test31-1/football_2-31.msgpack",1.80,-3.3),
    ("test31-1/football_3-31.msgpack",1.80,-3.3),
    ("test31-1/adults-31.msgpack",1.80,-3.3),
    ("fietsen-20.msgpack",2.0,-2.6),
    ("fietsen2-20.msgpack",2.0,-2.6),
    # ("test31-1/one_at_a_time-31.msgpack", 1.80, -3.3),
    # ("fietsen4-20.msgpack", 2.0, -2.6),
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

    featurevecs = np.zeros((11))

    x, y = pol2cart(data[:,0], data[:,1])

    _, elevation = pol2cart(data[:,0], data[:,4]+(3.1415/180*angle))
    elevation += height
    featurevecs[0] = averaged[0] #range
    featurevecs[1] = averaged[1] #angle
    featurevecs[2] = averaged[2] #doppler
    #featurevecs[3] = np.mean(elevation) #height
    featurevecs[4] = averaged[3] #snr

    featurevecs[4] = np.var(x)#deviation[0]
    featurevecs[5] = np.var(y)#deviation[1]
    #featurevecs[7] = deviation[2]
    #featurevecs[8] = deviation[3]

    featurevecs[6] = np.percentile(elevation, 95)
    featurevecs[7:10] = variance[0:3]

    #featurevecs[10] = np.percentile(elevation, 5)
    #featurevecs[15] = np.mean(data[:,3]) / ((1/(averaged[0]/1400) +130)) if averaged[0] > 6 else (np.mean(data[:,3])/360)
    #featurevecs[7] = averaged[3]
    #featurevecs[16] = summed[3]
    #featurevecs[17] = points
    #Out: [num points, range, angle, doppler, snr tot, snr avg, angle stdev, doppler stdev, rangedev, snr stdev ]

    return featurevecs

featurevector_length= 11

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

def countPointclouds(pcs):
    count = 0
    for pc in pcs:
        if(pc.size > 0):
            count += 1
    return count

numpointsinclouds = []
numpointcloudsintrace = []
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
        if countPointclouds(pointclouds)>=100:
            numpointcloudsintrace.append(1)
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

# def showHeight(features, labels, sel):
# #     dataframe = pd.DataFrame(np.hstack((np.expand_dims(labels, axis=1), features)),
# #                              columns= ["label", "range", "angle", "doppler", "stdx", "stdy", "dopplerstd", "height95", "height5"])
# #     dataframe = dataframe[["label","height95"]]
# #     grouped = dataframe.groupby('label',axis=0)
# #     print(grouped)
# #     #grouped.boxplot(subplots=False)
# #     #plt.show()
# #     import seaborn as sns
# #     sns.set()
# #     sns.boxplot(x="label", y="height95", data=dataframe)
# #     plt.gca().set_xticklabels("adult,bicycle,child".split(','))
# #     plt.ylim(.5,2.5)
# #     plt.xlabel("Class")
# #     plt.ylabel("height [m]")
# #     plt.title("95th percentile point height for each class")
# #     plt.show()

def showDifference(features, labels, a):
    dataframe = pd.DataFrame(np.hstack((np.expand_dims(labels, axis=1), features)),
                             columns= ["label", "range", "angle", "doppler", "stdx", "stdy", "dopplerstd", "height95", "height5","SNR"])

    dataframe = dataframe[dataframe['label']<3.1]
    labelnames = ["adult","bicycle", "child"]
    dataframe['label'] = dataframe['label'].apply(lambda x:labelnames[int(x)])
    dataframe = dataframe[abs(dataframe['angle'])<1]
    dataframe = dataframe[abs(dataframe['doppler']) > 1]

    sns.set()
    sns.scatterplot("range", "height95", data=dataframe,hue='label',size=1)
    plt.title("SNR vs Range for 3 classes")
    #
    # sns.jointplot("range", "SNR", data=dataframe[dataframe['label']==2],kind="kde")
    # sns.jointplot("range", "SNR", data=dataframe[dataframe['label']==1],kind="kde")
    # sns.jointplot("range", "SNR", data=dataframe[dataframe['label']==0],kind="kde")
    plt.show()


def printNumLabels(val_labels):
    unique, counts = np.unique(val_labels, return_counts=True)
    print(dict(zip(unique, counts)))

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
        plt.xlim(0,40)
        plt.xlabel("number of points")
        plt.ylabel("normalized samples")
        plt.title("Distribution of number of points in a cluster")
        plt.show()

    #print(np.sum(np.array(numpointcloudsintrace)))

    #distributionOfNumPoints()
    #exit()
    printNumLabels(labels)

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
    printNumLabels(val_labels)

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
    #clf = RandomForestClassifier(min_impurity_decrease=0.01, criterion="entropy", random_state=0,n_estimators=50)
    #clf = nn.MLPClassifier((50,50), max_iter=250, alpha=0.1,early_stopping=True)
    clf = svm.SVC(kernel='rbf',C=1, gamma=0.1,probability=True)
    #clf = load('randomForrest.joblib')
    clf.fit(train_features,train_labels)
    #dump(clf, 'randomForrest.joblib')
    #print(a.shape, np.argmax(res.numpy(),axis=1).shape)

    #print("feature importance: ", clf.feature_importances_)
    # print("Train:")
    # plot_confusion_matrix(train_labels, clf.predict(train_features), [ 'adult', 'bicycle','child', 'unlabled'])
    #
    validation_pred = clf.predict(val_features)
    #print(sum([x.tree_.node_count for x in clf.estimators_]))

    print("val:")
    print(np.mean(validation_pred == val_labels))
    from statsmodels.stats import proportion
    # L,H = proportion.proportion_confint(np.sum(validation_pred == val_labels), len(validation_pred), method='wilson')
    # print(L,H)
    plot_confusion_matrix(val_labels, validation_pred, ['adult', 'bicycle', 'child','unlabeled'], normalize=True)
    #plt.title("Confusion matrix, SVM (RBF kernel)")
    #plt.show()

    # res_acc.append(np.mean(validation_pred == val_labels))
    # res_L.append(L)
    # res_H.append(H)
    probs = clf.predict_proba(val_features)
    #
    uids = np.unique(val_ids)
    for averageamount in range(10,11):
        results = []
        test = 0
        for uid in uids:
            thisuid = probs[val_ids == uid]
            test += max(0,thisuid.shape[0]-(averageamount-1))
            i = 0
            while (i + averageamount) <= thisuid.shape[0]:
                prediction = np.mean(thisuid[i:i+averageamount], axis=0)
                result = np.argmax(prediction) == val_labels[val_ids == uid][0]
                results.append(result)
                i+= 1
            #print(f"uid: {uid}, predict: {prediction}, options:{np.sum(val_ids == uid)}, true = {val_labels[val_ids==uid][0]}")

        #print(test)
        #print(len(results))
        interval = proportion.proportion_confint(np.sum(results), len(results), method='wilson')
        print(averageamount,',', np.mean(results),',',interval[0],',',interval[1],',',len(results))

# for i in range(1,50,2):
#     print("===============")
#     print(i)
#     res_num.append(i)
doTheThing(10)

# print(res_num)
# # print(res_acc)
# # print(res_L)
# # print(res_H)