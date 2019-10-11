from lib.frameParser import Cluster, ParsedFrame
from PySide2 import QtCore
import numpy as np


class TIDError(Exception):
    def __init__(self):
        super().__init__()

class POI():
    counter = 0

    def __init__(self, tid):
        self.tid = tid  # TID assigned by sensor
        self.pointclouds = [] # Collection of point clouds over time.
        self.track = []  # Movement over time
        self.class_id = 0
        self.uid = POI.counter
        POI.counter += 1

    def addCluster(self, cluster: Cluster):
        if(cluster.tid != self.tid):
            raise TIDError()
        self.pointclouds.append(cluster.getPoints())
        self.track.append(np.array([cluster.info['posx'], cluster.info['posy']], dtype=float))

class POITracker(QtCore.QObject):

    def __init__(self):
        super().__init__()
        self.activePOIs = []
        self.history = []

    def processFrame(self, frame: ParsedFrame):

        #Remove POI's no longer in frame
        tids = [x.tid for x in frame.clusters]
        for poi in self.activePOIs:
            if poi.tid not in tids:
                self.activePOIs.remove(poi)
                self.history.append(poi)


        #For each cluster, find if we are already tracking that TID
        for cluster in frame.clusters:
            foundPOI = False
            for poi in self.activePOIs:
                if poi.tid == cluster.tid:
                    poi.addCluster(cluster)
                    foundPOI = True
            #If the TID is not found, add it
            if foundPOI == False:
                newPOI = POI(cluster.tid)
                newPOI.addCluster(cluster)
                self.activePOIs.append(newPOI)

    def getLocations(self):
        locations = []
        for poi in self.activePOIs:
            locations.append(poi.track[-1])
        return np.ascontiguousarray(np.array(locations))







