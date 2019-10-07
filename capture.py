import sys
import io
import pyqtgraph as pg
import numpy as np
import serial
import h5py
from datetime import datetime
import time
import joblib
import sklearn
import classifier
import lib.parser as parser
import msgpack

# commandport = "/dev/ttyACM0"
# dataport = "/dev/ttyACM1"
commandport = "COM10"
dataport = "COM11"

datasetName = "pointclouds"

#Label for new samples

#Packet definition


colormap = [[1.0,0.0,0.0,0.8],[1.0,1.0,0.0,.8],[0.0,1.0,0.0,.8],[0.0,1.0,1.0,.8],[1.0,0.0,1.0,.8]]

labels = []
datasamples = []
currentsample = []
packetsinsample = 0

scaler = joblib.load('scaler.joblib')
#model = joblib.load('model_logistic.joblib')
#model = joblib.load('model_norandom.joblib')
#model = joblib.load('model_logistic.joblib')
model = joblib.load('model_svc.joblib')
classes = ["child", "adult", "bicyclist"]#, "random"]
n_classes = len(classes)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)




def startSensor():
    with serial.Serial(commandport,115200, parity=serial.PARITY_NONE) as controlSerial:
        with open("customchirp.cfg", 'r') as configfile:
            for line in configfile:
                print(">> " + line, flush = True)
                controlSerial.write(line.encode('ascii'))
                print("<< " + controlSerial.readline().decode('ascii'), flush = True) #echo
                print("<< " + controlSerial.readline().decode('ascii'), flush = True) #"done"
                controlSerial.read(11) #prompt
                time.sleep(0.01)
            print("sensor started")

def addSample(data):
    """
        Add a sample of shape:
        4xn: (range, azimuth, doppler, snr) * numpoints

    """
    out = np.zeros([1,50, 4])
    for i, d in enumerate(data):
        if(i < 50):
            out[0,i] = [d['range'], d['angle'], d['doppler'], d['snr']]
    writeDataset(samples, out)
    #writeElement(labels, label)
    writeElement(timestamps, datetime.timestamp(datetime.now()))

            
def matchArrays(a, b):
    if len(a) != len(b):
        return False
    
    for i in range(0, len(a)):
        if a[i] != b[i]:
            return False
    return True 

                    
#Tracking and predicting
max_targets = 15
target_samples = [0] * max_targets
predictions = np.ones((max_targets,n_classes,20))/n_classes #[max_targets,3,10] tensor
def addPrediction(id, prediction):
    for i in range(19,0, -1):
        predictions[id, :,i] = predictions[id,:,i-1]
    predictions[id,:,0] = prediction
    return getPrediction(id)

def getPrediction(id):
    return np.mean(predictions[id, :, :], axis=1)


def predict_targets(frame):
    for tid in range(max_targets):
        tid_active = False
        for cluster in frame.clusters:
            if tid == cluster.tid:
               tid_active = True
               if(len(cluster.points) > 2):
                    points = np.array([cluster.getPoints()])
                    features = classifier.get_featurevector(points)
                    features = scaler.transform(features) #Batch norm
                    pred = model.predict_proba(features)
                    pred = addPrediction(tid, pred[0,:]) #LPF
                    target_samples[tid] += 1
                    classid = np.argmax(pred)
                    print(tid,classid, pred[classid], points.shape[1])
                    if target_samples[tid] > 3:
                        textitems[tid].setPos(cluster.info['posx'], cluster.info['posy'])
                        textitems[tid].setText(f"{int(pred[classid]*100):3}% {classes[classid]}",  _callSync='off')
        #if we didnt find the TID, then remove the id
        if tid_active == False and target_samples[tid] > 0:
            print(f"clear {tid}")
            textitems[tid].setText("",  _callSync='off')
            target_samples[tid] = 0

def visualizeFrame(frame):
    #Targets
    #print(frame.clusters)
    POIs = np.array([[cluster.info['posx'],cluster.info['posy']] for cluster in frame.clusters])
    scatter2.setData(pos=POIs) #, color = np.array([colormap[i['tid'] % (len(colormap)-1)] for i in packet['data']]))
    plot2.setData(POIs,  _callSync='off')

    #Point cloud
    points = []
    colors = []
    for cluster in frame.clusters:
        for point in cluster.points:
            y, x = pol2cart(point['range'], point['angle'])
            vel = point['doppler']
            colors.append(colormap[cluster.info['tid'] % (len(colormap)-1)])
            points.append(np.array([x,y,vel]))

    #unclustered points
    for point in frame.points:
        y, x = pol2cart(point['range'], point['angle'])
        vel = point['doppler']
        colors.append([0,0,0,.5])
        points.append(np.array([x, y, vel]))

    #[print(c.points, flush=True) for c in frame.clusters]
    points = np.array(points)
    scatterplot.setData(pos=points,color=np.array(colors), _callSync='off')

# SET UP GRAPHING

import pyqtgraph.multiprocess as mp

proc = mp.QtProcess(processRequests=False)
rpg = proc._import('pyqtgraph')
gl = proc._import('pyqtgraph.opengl')
view = gl.GLViewWidget()
view.show()
grid = gl.GLGridItem()
scatterplot = gl.GLScatterPlotItem()
scatter2 = gl.GLScatterPlotItem(color = [1.0,0,0,0.2] , size = 50)
#Draw the area we are viewing.
background = gl.GLLinePlotItem(pos=np.array([[-5,0,0],[5,0,0], [10,25,0], [-10,25,0],[-5,0,0]]),color=(1,1,1,1), width=2, antialias=True, mode='line_strip')

view.addItem(grid)
view.addItem(background)
view.addItem(scatterplot)
view.addItem(scatter2)


proc2 = mp.QtProcess(processRequests=False)
rpg2 = proc2._import('pyqtgraph')
plotwindow2 = rpg2.plot()
plot2 = plotwindow2.plot( pen=None, symbol='o')
plotwindow2.setRange(xRange=[-10,10], yRange=[0,25])

textitems = [rpg2.TextItem(text="test") for i in range(max_targets)]
for x in textitems:
    plotwindow2.addItem(x)

#END OF GRAPHING


#Open a file to store data
f = h5py.File(sys.argv[1]+datetime.now().strftime("%Y%m%d%H%M%S") + ".hdf5", 'w')
try:
    samples = f['/'+datasetName+'/samples']
    labels = f['/'+datasetName+'/labels']
    timestamps = f['/'+datasetName+'/timestamps']
except KeyError as e:
    samples = f.create_dataset('/'+datasetName+'/samples',(0, 50, 4), maxshape = (None, 50, 4))
    labels = f.create_dataset('/'+datasetName+'/labels',(0,), maxshape = (None,),chunks=True)
    timestamps = f.create_dataset('/'+datasetName+'/timestamps',(0,), maxshape = (None,),chunks=True)

def captureThreadMain(port):
    print("Start listening on COM11",flush = True)
    with serial.Serial(port, 921600) as dataSerial:
        buffer = []
        while(True):
            byte = dataSerial.read(1)
            #print(byte)
            buffer += byte
            #Check if we received full packet
            if matchArrays([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07], buffer[-8:]):
                #print(f"packet, size:{len(buffer)}", flush = True)
                frame = parser.parseFrame(bytes(buffer))
                if frame != None:
                    visualizeFrame(frame)
                    predict_targets(frame)
                buffer = buffer[-8:]

startSensor()
captureThreadMain(dataport)