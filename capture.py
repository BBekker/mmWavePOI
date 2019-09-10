import sys
import io
from construct import *
import pyqtgraph as pg
import numpy as np
import serial
import h5py
from datetime import datetime
import time
import joblib
import sklearn
import classifier

# commandport = "/dev/ttyACM0"
# dataport = "/dev/ttyACM1"
commandport = "COM10"
dataport = "COM11"

datasetName = "pointclouds"

#Label for new samples
label = sys.argv[2]

#Packet definition

rangeFFTSize = 256
dopplerFFTSize = 256
antennas = 8

colormap = [[1.0,0.0,0.0,0.5],[1.0,1.0,0.0,.5],[0.0,1.0,0.0,.5],[0.0,1.0,1.0,.5],[1.0,0.0,1.0,.5]]

labels = []
datasamples = []
currentsample = []
packetsinsample = 0

model = joblib.load('model.joblib')

from enum import Enum
class Message(Enum):
    MMWDEMO_OUTPUT_MSG_DETECTED_POINTS = 1
    MMWDEMO_OUTPUT_MSG_RANGE_PROFILE = 2 
    MMWDEMO_OUTPUT_MSG_NOISE_PROFILE = 3
    MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP = 4
    MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP = 5
    MMWDEMO_OUTPUT_MSG_POINT_CLOUD = 6
    MMWDEMO_OUTPUT_MSG_TARGET_LIST = 7
    MMWDEMO_OUTPUT_MSG_TARGET_INDEX = 8
    MMWDEMO_OUTPUT_MSG_STATS = 9
    MMWDEMO_OUTPUT_MSG_HEATMAP = 10
    MMWDEMO_OUTPUT_MSG_MAX = 11
    
    
frame = Aligned(4,
    Struct(
    #Find sync bytes by looping over untill we find the magic word
    "sync" / RepeatUntil(lambda x, lst, ctx : lst[-8:] == [0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07], Byte),
    "header" / Struct(
        "version" / Int32ul,
        'platform' / Int32ul, 
        'timestamp' / Int32ul,
        'totalPacketLen' / Int32ul, 
        'frameNumber' / Int32ul, 
        'subframeNumber' / Int32ul,
        'chirpProcessingMargin' / Int32ul, 
        'frameProcessingMargin' / Int32ul, 
        'trackingProcessingTime' / Int32ul,
        'uartSendingTime' / Int32ul,
        'numTLVs' / Int16ul, 
        'checksum' / Int16ul,
    ),
    "packets" / Struct(
             "type" / Int32ul,
             "len" / Int32ul,
             "data" / Switch(this.type,
                {
                Message.MMWDEMO_OUTPUT_MSG_POINT_CLOUD.value: 
                    "objects" / Struct(
                        "range" / Float32l,
                        "angle" / Float32l,
                        "doppler" / Float32l,
                        "snr" / Float32l,
                    )[lambda ctx: int((ctx.len - 8) / 16)],
                
                Message.MMWDEMO_OUTPUT_MSG_TARGET_LIST.value: 
                    "targets" / Struct(
                        "tid" / Int32ul,
                        "posx" / Float32l,
                        "posy" / Float32l,
                        "velX" / Float32l,
                        "velY" / Float32l,
                        "accX" / Float32l,
                        "accY"/ Float32l,
                        "ec" / Float32l[9],
                        "g" / Float32l
                    )[lambda ctx: int((ctx.len-8) / (17*4))],
                 
                Message.MMWDEMO_OUTPUT_MSG_TARGET_INDEX.value:
                    "indices" / Int8ul[this.len - 8],
                    
                Message.MMWDEMO_OUTPUT_MSG_NOISE_PROFILE.value: 
                    Array(rangeFFTSize, Int16ul),
                    
                Message.MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP.value: 
                    Array(rangeFFTSize * antennas, Struct("Img" / Int16sl, "Re" / Int16sl)),
                    
                Message.MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP.value: 
                    Array(rangeFFTSize * dopplerFFTSize, Int16ul),
                    
                Message.MMWDEMO_OUTPUT_MSG_STATS.value: 
                    Struct(
                    "interFrameProcessingTime" / Int32ul,
                    "transmitOutputTime" / Int32ul,
                    "interFrameProcessingMargin" / Int32ul,
                    "interChirpProcessingMargin" / Int32ul,
                    "activeFrameCPULoad" / Int32ul,
                    "interFrameCPULoad" / Int32ul
                ),

                Message.MMWDEMO_OUTPUT_MSG_HEATMAP.value:
                    Float32l[lambda ctx: int(ctx.len / 4)],
                }
                , default=Array(this.len, Byte))
         )[this.header.numTLVs] 
    )
)


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def parseData(stream, stopEvent):

    dataSerial = serial.Serial(stream, 921600)
    while True:
        buffer += dataSerial.read(1)
        outputfile.write(bytes(buffer[-1:]))
        if matchArrays([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07], buffer[-8:]):
            try:
                data = frame.parse(bytes(buffer))
                handleData(data)
            except StreamError:
                print("streamerror, streamsize: {}".format(buffer))
            buffer = buffer[-8:]


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
                parseThreadMain(bytes(buffer))
                buffer = buffer[-8:]
                    

previous = {"header":{"frameNumber": -1}}
def predict_targets(parsed):
    global previous
    #tracking lags one frame behind the pointcloud
    print(previous['header']['frameNumber'], parsed['header']['frameNumber'], flush=True)
    if previous['header']['frameNumber'] == parsed['header']['frameNumber'] - 1:

        if Message.MMWDEMO_OUTPUT_MSG_TARGET_LIST.value in parsed and Message.MMWDEMO_OUTPUT_MSG_POINT_CLOUD.value in previous:
            for target in parsed[Message.MMWDEMO_OUTPUT_MSG_TARGET_LIST.value]:
                tid = target['tid']
                #print(tid, parsed[Message.MMWDEMO_OUTPUT_MSG_POINT_CLOUD.value])
                points = [[d['range'], d['angle'], d['doppler'], d['snr']] for i, d in enumerate(previous[Message.MMWDEMO_OUTPUT_MSG_POINT_CLOUD.value]) if parsed[Message.MMWDEMO_OUTPUT_MSG_TARGET_INDEX.value][i] == tid]
                if(len(points) > 2):
                    points = np.array([points])
                    features = classifier.get_featurevector(points)
                    pred = model.predict_proba(features)
                    predid = np.argmax(pred)
                    print(tid,predid, pred[0,predid], points.shape[1])
    previous = parsed



def parseThreadMain(rawData):

    if(rawData == None):
        return
    try:
        data = frame.parse(rawData)
        parsed = {"header": data['header']}

        for packet in data['packets']:
            parsed[packet['type']] = packet['data']
        
            if packet['type'] == Message.MMWDEMO_OUTPUT_MSG_TARGET_LIST.value :
                POIs = np.array([[x['posy'],x['posx'] * -1] for x in packet['data']])
                scatter2.setData(pos=POIs, color = np.array([colormap[i['tid'] % (len(colormap)-1)] for i in packet['data']]))

            # if packet['type'] == Message.MMWDEMO_OUTPUT_MSG_HEATMAP.value:
            #     print("heatmap, len = {}".format(packet['len']/4), flush=True)
            #     heatmap.setImage(np.flip(np.reshape(packet['data'], (64,128)), 1))

            if packet['type'] == Message.MMWDEMO_OUTPUT_MSG_POINT_CLOUD.value:
                #print(f"pointcloud, points: {(packet['len'] - 8) / 16}", flush="True")
                x, y = pol2cart([x["range"] for x in packet['data']], [x['angle'] for x in packet['data']])
                vel = [x["doppler"] for x in packet['data']]
                colors = [(x['snr']/10,x['snr']/10,x['snr']/10) for x in packet['data']]     #Brightness of the point is SNR
                #pointcloud.setData([x["range"] for x in packet['data']], [x['angle'] for x in packet['data']])
                
                scatterplot.setData(pos=np.column_stack((x,y * -1,vel)),color=np.array(colors))

                addSample(packet['data'])
                # global currentsample, packetsinsample, datasamples
                # currentsample += packet['data']
                # packetsinsample += 1
                # if packetsinsample == 10 and len(currentsample) > 30:
                #     addSample(currentsample)
                #     currentsample = []
                #     packetsinsample = 0
                #     print('new sample')
                # elif packetsinsample > 10:
                #     #Too few points, probably nothing interesting to see
                #     currentsample = []
                #     packetsinsample = 0
        predict_targets(parsed)


    except StreamError:
        print("bad packet")

def writeDataset(dataset, data):
    dataset.resize(dataset.shape[0] + data.shape[0], axis=0)
    dataset[-data.shape[0]:] = data

def writeElement(dataset, data):
    dataset.resize(dataset.shape[0]+1,axis=0)
    dataset[-1] = data

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
background = gl.GLLinePlotItem(pos=np.array([[0,-10,0],[0,10,0], [25,10,0], [25,-10,0],[0,-10,0]]),color=(1,1,1,1), width=2, antialias=True, mode='line_strip')

view.addItem(grid)
view.addItem(background)
view.addItem(scatterplot)
view.addItem(scatter2)

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

#Send the startup commands to the sensor

startSensor()

#Start processing threads
#captureThread.start()
#writeThread.start()
#parseThread.start()

captureThreadMain(dataport)

