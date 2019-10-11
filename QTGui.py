import sys
import random
from PySide2 import QtCore, QtWidgets, QtGui,QtOpenGL
import pyqtgraph as pg
import pyqtgraph.opengl as pgl
import numpy as np
from lib.worker import Worker
import time
from lib import frameParser, classifier, storage
import serial
from lib.util import *
import joblib
from lib.POI import POITracker

commandport = "COM10"
dataport = "COM11"

max_targets = 15

class SerialReceiver(QtCore.QObject):

    frameReceived = QtCore.Signal(frameParser.ParsedFrame)

    def __init__(self):

        QtCore.QObject.__init__(self)
        self.serial = serial.Serial(dataport, 921600)
        self.buffer = []

    def run(self):
        while(self.serial.in_waiting > 0):
            byte = self.serial.read(1)
            # print(byte)
            self.buffer += byte
            # Check if we received full packet
            if frameParser.matchArrays([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07], self.buffer[-8:]):
                # print(f"packet, size:{len(self.buffer)}", flush = True)
                frame = frameParser.parseFrame(bytes(self.buffer))
                if frame != None:
                    #visualizeFrame(frame)
                    #predict_targets(frame)
                    #store_data(frame)
                    self.frameReceived.emit(frame)
                    #print("frame")
                self.buffer = self.buffer[-8:]

class Predictor(QtCore.QObject):
    newPrediction = QtCore.Signal(dict)

    def __init__(self, averaging):
        QtCore.QObject.__init__(self)

        ##Mod this for model
        self.scaler = joblib.load('scaler.joblib')
        # model = joblib.load('model_logistic.joblib')
        # model = joblib.load('model_norandom.joblib')
        # model = joblib.load('model_logistic.joblib')
        self.model = joblib.load('model_svc.joblib')
        self.classes = ["child", "adult", "bicyclist"]  # , "random"]
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

    def predict_targets(self, frame):
        #print("predicting...")
        result = {}
        for tid in range(self.max_targets):
            tid_active = False
            for cluster in frame.clusters:
                if tid == cluster.tid:
                    tid_active = True
                    if (len(cluster.points) > 2):
                        points = np.array([cluster.getPoints()])
                        features = classifier.get_featurevector(points)
                        features = self.scaler.transform(features)  # Batch norm
                        pred = self.model.predict_proba(features)
                        pred = self.addPrediction(tid, pred[0, :])  # LPF
                        self.target_samples[tid] += 1
                        classid = np.argmax(pred)
                        #print(tid, classid, pred[classid], points.shape[1])

                        result['x'] = cluster.info['posx']
                        result['y'] = cluster.info['posy']
                        result['tid'] = tid
                        result['result'] = pred
                        result['samples'] = self.target_samples[tid]
                        self.newPrediction.emit(result)
            # if we didnt find the TID, then remove the id
            if tid_active == False and self.target_samples[tid] > 0:
                result['tid'] = tid
                result['samples'] = 0
                self.newPrediction.emit(result)
                self.target_samples[tid] = 0

class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.threadpool = QtCore.QThreadPool()

        #Create basic layout
        self.button = QtWidgets.QPushButton("Start sensor")
        self.text = QtWidgets.QLabel("Hello World")
        self.text.setAlignment(QtCore.Qt.AlignCenter)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)

        self.button.clicked.connect(self.magicExecutor)

        #Set up graphs
        self.graphscontainer = QtWidgets.QHBoxLayout()

        #POI container
        self.POIs = np.random.rand(10, 2) * 30 - 15

        self.plotwindow = pg.PlotWidget()
        self.plot2 = self.plotwindow.plot(self.POIs, pen=None, symbol='o')
        self.plotwindow.setRange(xRange=[-10,10], yRange=[0,25])

        self.textitems = [pg.TextItem(text="") for i in range(max_targets)]
        for x in self.textitems:
            self.plotwindow.addItem(x)

        self.plotwindow.scene().sigMouseMoved.connect(self.mouseMoved)
        self.graphscontainer.addWidget(self.plotwindow)
        self.plotwindow.sizeHint = QtCore.QSize(100,100)


        self.glview = pgl.GLViewWidget()
        grid = pgl.GLGridItem()
        self.pointcloud = pgl.GLScatterPlotItem()
        self.POI3D = pgl.GLScatterPlotItem(color=[1.0, 0, 0, 0.2], size=50)
        # Draw the area we are viewing.
        background = pgl.GLLinePlotItem(pos=np.array([[-5, 0, 0], [5, 0, 0], [10, 25, 0], [-10, 25, 0], [-5, 0, 0]]),
                                       color=(1, 1, 1, 1), width=2, antialias=True, mode='line_strip')

        self.glview.addItem(grid)
        self.glview.addItem(background)
        self.glview.addItem(self.pointcloud)
        self.glview.addItem(self.POI3D)
        self.graphscontainer.addWidget(self.glview)

        self.glview.setSizePolicy(self.plotwindow.sizePolicy())

        self.layout.addLayout(self.graphscontainer)

        #POI tracker
        self.POITracker = POITracker()

        #neural net runner
        self.predictor = Predictor(20)
        self.predictor.newPrediction.connect(self.showPrediction)
        #start serial reader
        try:
            self.serialReceiver = SerialReceiver()
            self.serialReceiver.frameReceived.connect(self.POITracker.processFrame)
            self.serialReceiver.frameReceived.connect(self.visualizeFrame)
            self.serialReceiver.frameReceived.connect(lambda x: self.threadpool.start(Worker(lambda: self.predictor.predict_targets(x))))

            #Start timer for serial reading
            self.readtimer = QtCore.QTimer(self)
            self.readtimer.timeout.connect(self.serialReceiver.run)
            self.readtimer.start(5)
        except Exception as msg:
            print("serial not started!", msg)

    def keyPressEvent(self, event):
        print("test", event.text())

    def mouseMoved(self, event):
        if self.POIs.shape[0] > 0:
            pos = event
            localpoint = self.plotwindow.getViewBox().mapSceneToView(pos)
            self.text.setText(f"x:{localpoint.x()}, y:{localpoint.y()}")
            mpoint = np.array([localpoint.x(), localpoint.y()])

            distances = np.sqrt(np.sum((mpoint - self.POITracker.getLocations()) ** 2, axis=1))
            closest = np.argmin(distances)
            if(distances[closest] < 1.0):
                self.text.setText(f"mouse over point {closest}")


    def showPrediction(self, results):
        ti = self.textitems[results['tid']]
        if(results['samples'] > 2):
            pred = results['result']
            cid = np.argmax(pred)

            classes = ["child", "adult", "bicyclist"]  # , "random"]
            ti.setText(f"{int(pred[cid]*100):3}% {classes[cid]}")
            ti.setPos(results['x'], results['y'])
        else:
            ti.setText("")

    @QtCore.Slot()
    def visualizeFrame(self, frame):

        colormap = [[1.0,0.0,0.0,0.8],[1.0,1.0,0.0,.8],[0.0,1.0,0.0,.8],[0.0,1.0,1.0,.8],[1.0,0.0,1.0,.8]]

        # Targets
        # print(frame.clusters)
        self.POIs = self.POITracker.getLocations()
        if(self.POIs.shape[0] > 0):
            self.POI3D.setData(pos=self.POIs)  # , color = np.array([colormap[i['tid'] % (len(colormap)-1)] for i in packet['data']])).
            self.plot2.setData(self.POIs)


        # Point cloud
        points = [np.array([0.0,0.0,0.0])]
        colors = [[0.0,0.0,0.0,0.0]]
        for cluster in frame.clusters:
            for point in cluster.points:
                y, x = pol2cart(point['range'], point['angle'])
                vel = point['doppler']
                colors.append(colormap[cluster.info['tid'] % (len(colormap) - 1)])
                points.append(np.array([x, y, vel],dtype=np.float32))

        # unclustered points
        for point in frame.points:
            y, x = pol2cart(point['range'], point['angle'])
            vel = point['doppler']
            colors.append([.0, .0, .0, .5])
            points.append(np.array([x, y, vel],dtype=np.float32))

        # [print(c.points, flush=True) for c in frame.clusters]
        points = np.array(points,dtype=np.float32)
        self.pointcloud.setData(pos=points, color=np.array(colors, dtype=np.float32))

    def magicExecutor(self):
        self.threadpool.start(Worker(lambda: startSensor(commandport)))





if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = MyWidget()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec_())