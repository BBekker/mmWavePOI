import sys
import argparse
from PySide2 import QtCore, QtWidgets, QtGui,QtOpenGL
import pyqtgraph as pg
import pyqtgraph.opengl as pgl
import numpy as np
import msgpack
import msgpack_numpy
from joblib import dump, load


msgpack_numpy.patch()

max_targets = 15

classes = ["adult", "bicyclist","child", "unlabeled"]  # , "random"]
shortcuts = ['a','s','d']

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def countPointclouds(pcs):
    count = 0
    for pc in pcs:
        if(pc.size > 0):
            count += 1
    return count

class DataHandler(QtCore.QObject):

    def __init__(self, file):
        QtCore.QObject.__init__(self)
        self.file = file
        with open(file, 'rb') as file:
            unpacker = msgpack.Unpacker(file, raw=False)
            self.pois = list(unpacker)
            #print(self.pois[-1])
            self.frames = [[] for x in range(self.pois[-1]['lastFrame'])]

            #link pois to frames
            for poi in self.pois:
                lastframe = poi['lastFrame']
                firstframe = lastframe - len(poi['track'])
                if( countPointclouds(poi['pointclouds']) > 1):
                    for i in range(firstframe, lastframe, 1):
                        self.frames[i].append(poi) #store a reference for efficient lookup



    def getPOIs(self, frameNo):
        frame = self.frames[frameNo]
        #print(frameNo)
        locs = []
        for poi in frame:
            offset = frameNo - poi['lastFrame']
            #print(offset)
            locs.append(poi['track'][offset])
        if len(locs) > 0:
            return np.stack(locs, axis=0)
        return np.array([[]])

    def save(self):
        with open(self.file, "wb") as output:
            for poi in self.pois:
                output.write(msgpack.packb(poi, use_bin_type=True))

    def get_pointcloud(self,frameNo):
        frame = self.frames[frameNo]
        locs = []
        for poi in frame:
            offset = frameNo - poi['lastFrame']
            if(poi['pointclouds'][offset].size != 0):
                locs.append(poi['pointclouds'][offset])
            # else:
            #     locs.append(np.empty((,4)))

        return locs

    def get_classes(self,frameNo):
        frame = self.frames[frameNo]
        return [poi['class_id'] for poi in frame]

    def getFrames(self):
        return len(self.frames)

    def get_poi_in_frame(self, frame, poi):
        return self.frames[frame][poi]

    # def get_featurevector(data):
    #     """
    #      Data = [range, angle, doppler, snr]
    #     """
    #     # print(data)
    #     # points = np.sum((np.sum(data, axis=2) != 0), axis=1)
    #     points = data.shape[0]
    #
    #     summed = np.sum(data, axis=0)
    #     averaged = summed / points
    #     # deviation = np.std(data, axis=1)
    #
    #     featurevecs = np.zeros((10))
    #
    #     featurevecs[0] = points
    #     featurevecs[1] = averaged[0]
    #     featurevecs[2] = averaged[1]
    #     featurevecs[3] = averaged[2]
    #     featurevecs[4] = averaged[3]
    #     featurevecs[5] = summed[3]
    #
    #     featurevecs[6] = np.std(data[:, 1])
    #     featurevecs[7] = np.std(data[:, 2])
    #     featurevecs[8] = np.std(data[:, 0])
    #     featurevecs[9] = np.std(data[:, 3])
    #     # Out: [num points, range, angle, doppler, snr tot, snr avg, angle stdev, doppler stdev, rangedev, rssi stdev ]
    #
    #     return featurevecs
    #
    # def getPredictions(self, frameNo):
    #     pointclouds = self.get_pointcloud(frameNo)




class MyWidget(QtWidgets.QWidget):
    def __init__(self, outputfile, playback , inputfile):
        super().__init__()

        self.datahandler = DataHandler(inputfile)

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
        background = pgl.GLLinePlotItem(pos=np.array([[-5, 0, 0], [5, 0, 0], [20, 35, 0], [-20, 35, 0], [-5, 0, 0]]),
                                       color=(1, 1, 1, 1), width=2, antialias=True, mode='line_strip')

        self.glview.addItem(grid)
        self.glview.addItem(background)
        self.glview.addItem(self.pointcloud)
        self.glview.addItem(self.POI3D)
        self.graphscontainer.addWidget(self.glview)

        self.glview.setSizePolicy(self.plotwindow.sizePolicy())

        self.layout.addLayout(self.graphscontainer)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(lambda : self.visualizeFrame(self.slider.value()))
        self.slider.setMaximum(self.datahandler.getFrames())

        self.timer = QtCore.QTimer()
        self.timer.setInterval(100);
        self.timer.timeout.connect(lambda : self.slider.setValue(self.slider.value()+1)) #incrementing the slider changes the frame displayed

        self.spinbox = QtWidgets.QSpinBox()
        self.spinbox.valueChanged.connect(lambda : self.timer.stop() if self.spinbox.value() == 0 else (self.timer.setInterval(1000//self.spinbox.value()), self.timer.start()))

        self.savebutton = QtWidgets.QPushButton("save")
        self.savebutton.clicked.connect(self.datahandler.save)

        self.controlLayout = QtWidgets.QHBoxLayout()
        self.controlLayout.addWidget(self.slider)
        self.controlLayout.addWidget(self.spinbox)
        self.controlLayout.addWidget(self.savebutton)

        self.layout.addLayout(self.controlLayout)

        #predictor:
        self.model = load('randomForrest.joblib')



    def keyPressEvent(self, event):
        #print("test", event.key())
        if event.text() in shortcuts:
            mpoint = np.array([self.mouseLocation.x(), self.mouseLocation.y()])
            frame = self.slider.value()
            distances = np.sqrt(np.sum((mpoint - self.datahandler.getPOIs(frame)) ** 2, axis=1))
            closest = np.argmin(distances)

            selected_class = shortcuts.index(event.text())

            if (distances[closest] < 1.0):
                poi = self.datahandler.get_poi_in_frame(frame, closest)
                self.text.setText(f"mouse over point {poi['uid']} = {classes[selected_class]}, was: {classes[poi['class_id']]}")
                poi['class_id'] = selected_class
            #self.POITracker.activePOIs[closest].class_id = selected_class

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


    def visualizeFrame(self, frame):

        colormap = [[1.0,0.0,0.0,0.8],[1.0,1.0,0.0,.8],[0.0,1.0,0.0,.8],[0.0,1.0,1.0,.8],[1.0,0.0,1.0,.8]]

        # Targets
        # print(frame.clusters)
        pois = self.datahandler.getPOIs(frame)
        if(pois.shape[0] > 0):
            self.POI3D.setData(pos=pois)  # , color = np.array([colormap[i['tid'] % (len(colormap)-1)] for i in packet['data']])).
        self.plot2.setData(pois)

        class_ids = self.datahandler.get_classes(frame)
        for i in range(len(self.textitems)):
            if pois.shape[0] > i:
                self.textitems[i].setPos(pois[i,0], pois[i,1])
                self.textitems[i].setText(f"{self.datahandler.get_poi_in_frame(frame,i)['uid']}: {classes[class_ids[i]]}")
            else:
                self.textitems[i].setText("")

    # try:
        # Point cloud
        colors = [[0.0,0.0,0.0,0.0]]
        pointclouds = self.datahandler.get_pointcloud(frame)
        pointclouds = np.concatenate(pointclouds, axis=0)

        y, x = pol2cart(pointclouds[:,0], pointclouds[:,1])
        _, vel = pol2cart(pointclouds[:,0], pointclouds[:,4])
        #print(y,x,pointclouds[:,3])
        poss = np.stack((x,y,vel+2.0),axis=1)
        self.pointcloud.setData(pos=poss)#, color=np.array(colors, dtype=np.float32))
        # except Exception as e:
        #     pass

def parse_arguments(args):
    parser = argparse.ArgumentParser(description='readout radar sensor')
    parser.add_argument('--file', help = "filename to store/read")
    parser.add_argument('--playback', help = "play back file" )
    #parser.add_argument('--no_raw', action='store_', help = "dont store raw file")
    return parser.parse_args()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    args = parse_arguments(sys.argv)

    widget = MyWidget(args.file, args.playback, args.playback)
    widget.resize(800, 600)
    widget.show()


    sys.exit(app.exec_())