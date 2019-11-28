from lib.frameParser import Cluster, ParsedFrame
from PySide2 import QtCore
import numpy as np
import msgpack
import msgpack_numpy
from datetime import datetime

msgpack_numpy.patch()

class TIDError(Exception):
    def __init__(self):
        super().__init__()


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
            self.history.append(poi)
            self.savepoi(poi)
            print("archived a poi")

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
