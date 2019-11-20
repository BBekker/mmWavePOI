import Torch.nn
import Torch.optim
import RNNTorch 


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
    #featurevecs[:,6] = deviation[:,1]
    #featurevecs[:,7] = deviation[:,2]

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


learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def main():
    datafile1 = h5py.File(sys.argv[1],'r')
    datafile2 = h5py.File(sys.argv[2], 'r')

    pc1 = datafile1['pointclouds/samples'][:3500]
    pc2 = datafile2['pointclouds/samples'][:]

    numa = pc1.shape[0]
    numb = pc2.shape[0]
    
    samples = np.concatenate((pc1, pc2))
    
    featurevecs = get_featurevector(samples)

    
    #print(featurevecs)
    labels = np.array(([0] * numa) + ([1] * numb))
    


    indices = np.arange(len(labels))

    # np.random.seed(13317)
    # np.random.shuffle(indices)
    print(f"datasets: {numa} {numb}")
    #set up SVM

    #Normalize our input values
    scaler = StandardScaler()
    scaler.fit(featurevecs)
    featurevecs = scaler.transform(featurevecs)

    #split up data
    
    trainset = featurevecs[indices[:-500],:]
    trainlabels = labels[indices[:-500]]

    testset = featurevecs[indices[-500:],:]
    testlabels = labels[indices[-500:]]

    






if __name__ == "__main__":
    main()