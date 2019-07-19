import os
import io
from construct import *
import pyqtgraph as pg
import numpy as np
import serial
import time
import datetime


#Packet definition

rangeFFTSize = 256
dopplerFFTSize = 256
antennas = 8

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
                        "g" / Float32l,
                        "heatmap" / Float32l[100]
                    )[lambda ctx: int((ctx.len-8) / (117*4))],
                 
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
    with serial.Serial("COM10",115200, parity=serial.PARITY_NONE) as controlSerial:
        with open("customchirp.cfg", 'r') as configfile:
            for line in configfile:
                print(">> " + line, flush = True)
                controlSerial.write(line.encode('ascii'))
                print("<< " + controlSerial.readline().decode('ascii'), flush = True) #echo
                print("<< " + controlSerial.readline().decode('ascii'), flush = True) #"done"
                controlSerial.read(11) #prompt
                time.sleep(0.01)
            print("sensor started")

            
def matchArrays(a, b):
    if len(a) != len(b):
        return False
    
    for i in range(0, len(a)):
        if a[i] != b[i]:
            return False
    return True 

def storeThreadMain(inputqueue,stopEvent):
    with open("output.bin", "wb") as outputfile:
        while(False):
            packet = inputqueue.get(True, 1000)
            outputfile.write(packet)

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
                print(f"packet, size:{len(buffer)}", flush = True)
                parseThreadMain(bytes(buffer))
                buffer = buffer[-8:]
                    
def parseThreadMain(rawData):

    if(rawData == None):
        return
    try:
        data = frame.parse(rawData)
        print("Packet:", flush=True)
        for packet in data['packets']:
            print("- type: {}".format(packet['type']))
            # if packet['type'] == Message.MMWDEMO_OUTPUT_MSG_TARGET_LIST.value :
            #     #print(packet['data'][0]['heatmap'])
            #     targets = {}
            #     for target in packet['data']:
            #         target['timestamp'] = datetime.datetime.now()
            #         targets[target['tid']] = target

            #     #print("location x:{} y:{}".format(target['posx'], target['posy']))
            #     if(len(targets) > 1):
            #         image = np.zeros([400, 400])
            #         for target in targets:

            #             x = targets[target]['posx']
            #             y = targets[target]['posy']
            #             xindex = int((x+10) / 35 * 380 + 10)
            #             yindex = int(y / 35 * 380 + 10)
            #             a = np.reshape(targets[target]['heatmap'],(10,10))
            #             print("x:{} y:{}".format(xindex, yindex), flush=True)
            #             image[xindex-5:xindex+5, yindex-5:yindex+5] = a
            #             #imgView.setImage(image, autoRange=False, autoLevels=False)
            #         print("show", flush=True)
            #         #QtGui.QApplication.processEvents()
            # if packet['type'] == Message.MMWDEMO_OUTPUT_MSG_HEATMAP.value:
            #     print("heatmap, len = {}".format(packet['len']/4), flush=True)
            #     heatmap.setImage(np.flip(np.reshape(packet['data'], (64,128)), 1))

            if packet['type'] == Message.MMWDEMO_OUTPUT_MSG_POINT_CLOUD.value:
                print(f"pointcloud, points: {(packet['len'] - 8) / 16}", flush="True")
                x, y = pol2cart([x["range"] for x in packet['data']], [x['angle'] for x in packet['data']])
                vel = [x["doppler"] for x in packet['data']]
                colors = [np.tanh([x['snr']/10,x['snr']/10,x['snr']/10,10000000]) for x in packet['data']]     #Brightness of the point is SNR
                #pointcloud.setData([x["range"] for x in packet['data']], [x['angle'] for x in packet['data']])
                scatterplot.setData(pos=np.column_stack((x,y,vel)),color=np.array(colors))


    except StreamError:
        print("bad packet")

imgView = None


import pyqtgraph.multiprocess as mp
#pg.mkQApp()
proc = mp.QtProcess(processRequests=False)
rpg = proc._import('pyqtgraph')
gl = proc._import('pyqtgraph.opengl')
#plotwin = rpg.plot()
#imgView = rpg.show(np.random.rand(500,500))
#heatmap = rpg.show(np.zeros((64,128)))
view = gl.GLViewWidget()
view.show()
grid = gl.GLGridItem()
scatterplot = gl.GLScatterPlotItem()
#Draw the area we are viewing.
background = gl.GLLinePlotItem(pos=np.array([[0,-10,0],[0,10,0], [25,10,0], [25,-10,0],[0,-10,0]]),color=(1,1,1,1), width=2, antialias=True, mode='line_strip')

view.addItem(grid)
view.addItem(background)
view.addItem(scatterplot)

#Send the startup commands to the sensor
startSensor()

#Start processing threads
#captureThread.start()
#writeThread.start()
#parseThread.start()

captureThreadMain("COM11")

