import sys
import argparse
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
from lib.POI import POITracker, Predictor
from datetime import datetime
import construct
import gtrack.gtrack as gtrack

commandport = "COM5"
dataport = "COM6"

max_targets = 15

classes = ["adult", "bicyclist","child"]  # , "random"]
shortcuts = ['a','s','d']

class SerialReceiver(QtCore.QObject):

    frameReceived = QtCore.Signal(frameParser.ParsedFrame)

    def __init__(self):

        QtCore.QObject.__init__(self)
        self.serial = serial.Serial(dataport, 921600)
        self.buffer = []

    def run(self):
        while(self.serial.in_waiting > 0):
            byte = self.serial.read(self.serial.in_waiting)
            bytes_read = len(byte)
            # Store raw data
            rawfile.write(byte)
            # print(byte)
            self.buffer += byte
            # Check if we received full packet
            for i in range(len(self.buffer)):
                if frameParser.matchArrays([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07], self.buffer[i:i+8]):
                    framelen = int.from_bytes(self.buffer[i+12+8:i+15+8], 'little')
                    if len(self.buffer) < i+framelen+1:
                        return

                    #print(f"packet, at i={i} packet len: {framelen} buffer size:{len(self.buffer)}", flush = True)
                    #print(self.buffer)
                    #print(self.buffer[i:i+framelen+1])
                    frame = frameParser.parseFrame(bytes(self.buffer[i:]))


                    if frame is not None:

                        #print("frame received")
                        self.frameReceived.emit(frame)
                    else:
                        print("parse failure")

                    self.buffer = self.buffer[i+framelen:]
                    break

class Playback(QtCore.QObject):

    frameReceived = QtCore.Signal(frameParser.ParsedFrame)

    def __init__(self, file, recluster):
        QtCore.QObject.__init__(self)
        self.file = file
        self.recluster = recluster
        if recluster:
            self.gtrack = gtrack.gtrack()

    def reclusterFrame(self, data):
        #print(data)
        pointCloudLocation = frameParser.getPacket(data, frameParser.Message.POINT_CLOUD)
        sideInfo = frameParser.getPacket(data, frameParser.Message.POINT_CLOUD_SIDE_INFO)
        pointCloud = [{**pointCloudLocation[i], **sideInfo[i]} for i in range(len(pointCloudLocation))]
        npdata = np.array([np.array([x['range'], x['azimuth'], x['doppler'], x['snr']]) for x in pointCloud])
        self.gtrack.step(npdata)
        #gtrack expects: [range, angle, doppler, SNR]
        print(self.gtrack.getData().numtracks)
        
        gtrackData = self.gtrack.getData()
        
        #Create a nice frame object
        parsedFrame = frameParser.ParsedFrame(data['header']['frameNumber'])
        for t in range(gtrackData.numtracks):
            target = self.gtrack.getTrack(t)
            #print(target.uid)
            tid =  target.uid
            cluster = parsedFrame.newCluster(tid)
            cluster.addInfo({'posx':target.S[0], 
                            'posy':target.S[1],
                            'sx':target.dim[0],
                            'sy':target.dim[1], 
                            'tid':target.tid})
            for targetIndex in range(len(pointCloud)):
                if gtrackData.indices[targetIndex] == tid:
                    cluster.addPoint(pointCloud[targetIndex])

        
        unique, counts = np.unique([gtrackData.indices[i] for i in range(len(pointCloud))], return_counts=True)
        print(dict(zip(unique, counts)))

        #Add points without a cluster
        for targetIndex in range(len(pointCloud)):
            if gtrackData.indices[targetIndex] >= 250:
                parsedFrame.addPoint(pointCloud[targetIndex])
        return parsedFrame



    def getFrame(self):
        try:
            parsed = construct.RawCopy(frameParser.frameParser).parse_stream(self.file)
        
            print(f"Getframe {parsed['offset2']} {self.file.tell()}",flush=True)
            self.file.seek(parsed['offset2'])
            
            #print(parsed)
            #print("=================")

            if self.recluster:
                frame = self.reclusterFrame(parsed['value'])
            else:
                frame = frameParser.parseFrame(parsed['data'])

            self.frameReceived.emit(frame)
        except Exception as e:
            print(e)
            self.file.seek(10,1) #probably a file error, skip a bit



class MyWidget(QtWidgets.QWidget):
    def __init__(self, outputfile, playback , inputfile, recluster):
        super().__init__()
        self.playback = playback

        self.threadpool = QtCore.QThreadPool()
        self.mouseLocation = None

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


        mesh = pgl.MeshData.sphere(3,5)
        self.clusterMesh = [pgl.GLMeshItem(meshdata=mesh, edgeColor=[.2,.9,.4,.1] ,drawEdges=True, drawFaces=False) for i in range(20)]
        [self.glview.addItem(mesh) for mesh in self.clusterMesh]
        self.glview.addItem(grid)
        self.glview.addItem(background)
        self.glview.addItem(self.pointcloud)
        #self.glview.addItem(self.POI3D)
        self.graphscontainer.addWidget(self.glview)

        self.glview.setSizePolicy(self.plotwindow.sizePolicy())

        self.layout.addLayout(self.graphscontainer)

        #POI tracker
        self.POITracker = POITracker(outputfile)

        #neural net runner
        #self.predictor = Predictor(20, max_targets)
        #self.predictor.newPrediction.connect(self.showPrediction)
        #start serial reader
        try:
            if (not self.playback):
                self.serialReceiver = SerialReceiver()
                self.serialReceiver.frameReceived.connect(self.POITracker.processFrame)
                self.serialReceiver.frameReceived.connect(self.visualizeFrame)     #Start timer for serial reading
                self.readtimer = QtCore.QTimer(self)
                self.readtimer.timeout.connect(self.serialReceiver.run)
                self.readtimer.start(5)
            else:
                self.playback = Playback(inputfile, recluster)
                self.playback.frameReceived.connect(self.POITracker.processFrame)
                self.playback.frameReceived.connect(self.visualizeFrame)
                self.readtimer = QtCore.QTimer(self)
                self.readtimer.timeout.connect(self.playback.getFrame)
                self.readtimer.start(100)
                print(" start playback")


        except Exception as msg:
            print("serial not started!", msg)



    def keyPressEvent(self, event):
        #print("test", event.key())
        mpoint = np.array([self.mouseLocation.x(), self.mouseLocation.y()])

        distances = np.sqrt(np.sum((mpoint - self.POITracker.getLocations()) ** 2, axis=1))
        closest = np.argmin(distances)

        if(event.text() == " "):
            self.POITracker.breakuppoi(self.POITracker.activePOIs[closest])

        try:
            selected_class = shortcuts.index(event.text())
            if (distances[closest] < 1.0):
                self.text.setText(f"mouse over point {self.POITracker.activePOIs[closest].uid} = {selected_class}")
            self.POITracker.activePOIs[closest].class_id = selected_class
        except ValueError as v:
            pass

    def mouseMoved(self, event):
        if self.POIs.shape[0] > 0:
            pos = event
            localpoint = self.plotwindow.getViewBox().mapSceneToView(pos)
            self.mouseLocation = localpoint


    def showPrediction(self, results):
        ti = self.textitems[results['tid']]
        if(results['samples'] > 2):
            pred = results['result']
            cid = np.argmax(pred)


            ti.setText(f"{results['tid']}: {int(pred[cid]*100):3}% {classes[cid]} set to: {self.POITracker.poiFromTid(results['tid']).class_id}")
            ti.setPos(results['x'], results['y'])
        else:
            ti.setText("")

    @QtCore.Slot()
    def visualizeFrame(self, frame):

        colormap = [[1.0,0.0,0.0,0.8],[1.0,1.0,0.0,.8],[0.0,1.0,0.0,.8],[0.0,1.0,1.0,.8],[1.0,0.0,1.0,.8]]

        # Targets
        # print(frame.clusters)
        self.POIs = self.POITracker.getLocations()
        pois = self.POITracker.getPOIs()
        if args.clustering:
            for i in range(20):
                if(len(frame.clusters) > i):
                    cluster = frame.clusters[i]
                    self.clusterMesh[i].resetTransform()
                    self.clusterMesh[i].setColor(colormap[cluster.info['tid'] % (len(colormap) - 1)])
                    self.clusterMesh[i].scale(cluster.info['sx'],cluster.info['sy'],1)
                    self.clusterMesh[i].translate(cluster.info['posx'],cluster.info['posy'],0)  # , color = np.array([colormap[i['tid'] % (len(colormap)-1)] for i in packet['data']])).
                else:
                    self.clusterMesh[i].resetTransform()
            self.plot2.setData(self.POIs)

        for i in range(len(self.textitems)):
            if len(pois) > i:
                pos = pois[i].getPos()
                self.textitems[i].setPos(pos[0], pos[1])
                self.textitems[i].setText(f"{pois[i].class_id}, h={pois[i].getHeight(90):.2f}")
            else:
                self.textitems[i].setText("")

        # Point cloud
        points = [np.array([0.0,0.0,0.0])]
        colors = [[0.0,0.0,0.0,0.0]]
        for cluster in frame.clusters:
            for point in cluster.points:
                y, x = pol2cart(point['range'], point['azimuth'])
                _, vel = pol2cart(point['range'], point['elevation'])
                colors.append(colormap[cluster.info['tid'] % (len(colormap) - 1)])
                points.append(np.array([x, y, vel],dtype=np.float32))

        # unclustered points
        #print(f"{len(frame.points)} unclustered points")
        for point in frame.points:
            y, x = pol2cart(point['range'], point['azimuth'])
            _, vel = pol2cart(point['range'], point['elevation'])
            colors.append([1.0, 1.0, 1.0, point['snr']/30.0])
            points.append(np.array([x, y, vel],dtype=np.float32))

        # [print(c.points, flush=True) for c in frame.clusters]
        points = np.array(points,dtype=np.float32)
        self.pointcloud.setData(pos=points, color=np.array(colors, dtype=np.float32))

    def magicExecutor(self):
        self.threadpool.start(Worker(lambda: startSensor(commandport)))

def parse_arguments(args):
    parser = argparse.ArgumentParser(description='readout radar sensor')
    parser.add_argument('--file', help = "filename to store/read")
    parser.add_argument('--playback', help = "play back file" )
    parser.add_argument('--clustering', help = "enable reclustering of pointclouds", action='store_true')
    #parser.add_argument('--no_raw', action='store_', help = "dont store raw file")
    return parser.parse_args()

args = None

if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    args = parse_arguments(sys.argv)
    print(args)
    outputfile = None
    inputfile = None
    if(args.file is not None):
        outputfile = open(f"{args.file}-{datetime.now().day}.msgpack", "ab")

        rawfile = open(f"raw/{datetime.now().strftime('%y-%m-%d_%H%M')}.bin", "ab")
    if(args.playback is not None):
        inputfile = open(args.playback, 'rb')
        inputfile.seek(0,2)
        print(inputfile.tell())
        inputfile.seek(0,0)
        #inputfile.seek(355000,0)




    widget = MyWidget(outputfile, args.playback, inputfile, args.clustering)
    widget.resize(800, 600)
    widget.show()


    sys.exit(app.exec_())