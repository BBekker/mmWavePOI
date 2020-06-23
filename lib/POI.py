from lib.frameParser import Cluster, ParsedFrame
from PySide2 import QtCore
import numpy as np
import msgpack
import msgpack_numpy
from datetime import datetime
import joblib
import lib.util

msgpack_numpy.patch()

class TIDError(Exception):
    def __init__(self):
        super().__init__()

class prediction_state():

    def __init__(self):
        self.feature_history = [] # [90% percentile, numpoints]

class POI():
    counter = 0

    def __init__(self, tid):
        self.tid = tid  # TID assigned by sensor
        self.pointclouds = []  # Collection of point clouds over time.
        self.track = []  # Movement over time
        self.class_id = -1
        self.uid = POI.counter
        self.lastFrame = 0
        self.timestamp = str(datetime.now())
        POI.counter += 1


    def addCluster(self, cluster: Cluster):
        if (cluster.tid != self.tid):
            raise TIDError()
        self.pointclouds.append(cluster.getPoints())
        self.track.append(np.array([cluster.info['posx'], cluster.info['posy']], dtype=float))

    def getHeight(self, percentile):
        array = self.pointclouds[-1]
        height = -1
        if(array.shape[0] > 1):
            _, height = lib.util.pol2cart(array[:,0], array[:,4])
        return np.percentile(height, percentile)

    def getPos(self):
        return self.track[-1]


class POITracker(QtCore.QObject):

    def __init__(self, outputfile):
        super().__init__()
        self.activePOIs = []
        self.history = []
        self.lastFrame = 0
        self.outputfile = outputfile

    def processFrame(self, frame: ParsedFrame):
        currentFrame = frame.frameNumber

        # For each cluster, find if we are already tracking that TID
        for cluster in frame.clusters:
            foundPOI = False
            for poi in self.activePOIs:
                if poi.tid == cluster.tid:
                    poi.addCluster(cluster)
                    poi.lastFrame = currentFrame
                    foundPOI = True
            # If the TID is not found, add it
            if foundPOI == False:
                newPOI = POI(cluster.tid)
                newPOI.addCluster(cluster)
                newPOI.lastFrame = currentFrame
                self.activePOIs.append(newPOI)

        # Remove POI's no longer in frame
        tids = [x.tid for x in frame.clusters]
        for poi in self.activePOIs:
            if poi.tid not in tids:
                if (poi.lastFrame < (currentFrame - 20)):
                    self.activePOIs.remove(poi)
                    self.onArchive(poi)

    def onArchive(self, poi):
        if len(poi.pointclouds) > 5:  # Require at least a few samples
            #self.cleanup(poi)
            self.history.append(poi)
            self.savepoi(poi)
            print("archived a poi")

    def cleanup(self, poi):
        for i in range(0, -len(poi.pointclouds), -1):
            if(poi.pointclouds[i].size > 0):
                poi.pointclouds = poi.pointclouds[:i]
                poi.track = poi.track[:i]
                poi.lastframe += i


    def getLocations(self):
        locations = []
        for poi in self.activePOIs:
            locations.append(poi.track[-1])
        return np.ascontiguousarray(np.array(locations))

    def poiFromTid(self, tid):
        return next(filter(lambda x: x.tid == tid, self.activePOIs))

    def savepoi(self, poi):
        if(self.outputfile != None):
            self.outputfile.write(msgpack.packb(vars(poi), use_bin_type=True))

    def breakuppoi(self, poi):
        """"In case the clustering fails, we want to be able to break up a POI in time."""
        self.activePOIs.remove(poi)
        self.onArchive(poi)

    def getPOIs(self):
        return self.activePOIs

class Predictor(QtCore.QObject):
    newPrediction = QtCore.Signal(dict)

    def __init__(self, averaging, max_targets):
        QtCore.QObject.__init__(self)

        ##Mod this for model
        self.scaler = joblib.load('scaler.joblib')
        # model = joblib.load('model_logistic.joblib')
        # model = joblib.load('model_norandom.joblib')
        # model = joblib.load('model_logistic.joblib')
        self.model = joblib.load('randomForrest.joblib')
        self.classes = ["adult", "bicyclist", "child", "unlabled"]  # , "random"]
        self.n_classes = len(self.classes)


        self.averaging = averaging
        self.max_targets = max_targets
        self.target_samples = [0] * self.max_targets
        self.predictions = np.ones((self.max_targets, self.n_classes, averaging)) / self.n_classes  # [max_targets,3,10] tensor

    def addPrediction(self, id, prediction):
        for i in range(self.averaging-1, 0, -1):
            self.predictions[id, :, i] = self.predictions[id, :, i - 1]
        self.predictions[id, :, 0] = prediction
        return self.getPrediction(id)

    def getPrediction(self, id):
        return np.mean(self.predictions[id, :, :], axis=1)

    def predict(self, poi: POI):
        pass



