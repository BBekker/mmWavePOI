import os
import io
from construct import *
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from matplotlib import animation, rc
import numpy as np
import serial
import time
import datetime
import queue
import threading


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
    MMWDEMO_OUTPUT_MSG_MAX = 10
    
    
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
                }, default=Array(this.len, Byte))
         )[this.header.numTLVs] 
    )
)



def parseData(stream, stopEvent):
    while not stopEvent.is_set():
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
                print(">> " + line)
                controlSerial.write(line.encode('ascii'))
                print("<< " + controlSerial.readline().decode('ascii')) #echo
                print("<< " + controlSerial.readline().decode('ascii')) #"done"
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
        while( not stopEvent.is_set()):
            packet = inputqueue.get(True, 1000)
            outputfile.write(packet)

def captureThreadMain(dataSerial, onPacket):
    print("Start listening on COM11")
    buffer = []
    while(not stopEvent.is_set()):
        byte = dataSerial.read(1)
        #print(byte)
        buffer += byte
        #Check if we received full packet
        if matchArrays([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07], buffer[-8:]):
            print("packet")
            onPacket(bytes(buffer))
            buffer = buffer[-8:]
                    
def parseThreadMain(inputqueue, stopEvent):

    #targets plots
    fig, ax = plt.subplots(1, 5)
    while not stopEvent.is_set():
        rawData = inputqueue.get(True, 1000)
        if(rawData == None):
            return
        try:
            data = frame.parse(rawData)
            for packet in data['packets']:
                if packet['type'] == Message.MMWDEMO_OUTPUT_MSG_TARGET_LIST.value :
                    #print(packet['data'][0]['heatmap'])
                    targets = {}
                    for target in packet['data']:
                        target['timestamp'] = datetime.datetime.now()
                        targets[target['tid']] = target

                    #print("location x:{} y:{}".format(target['posx'], target['posy']))
                    if(len(targets) > 1):
                        images = []
                        fig.suptitle('Targets')
                        i = 0
                        normalizer = matplotlib.colors.Normalize()
                        for target in targets:
                            a = np.reshape(targets[target]['heatmap'],(10,10))

                            images.append(ax[i].imshow(a, cmap='hot', norm= normalizer, interpolation='nearest'))
                            #ax[i].setTitle("x:{} y:{}".format(targets[target]['posx'], targets[target]['posy']))
                            i += 1
                        fig.colorbar(images[0])
                            #plt.imshow(a, cmap='hot', norm= matplotlib.colors.Normalize(), interpolation='nearest')
                            #plt.show()
                            #print(a)
                            
                        plt.scatter([targets[x]['posx'] for x in targets], [targets[x]['posy'] for x in targets])
        except StreamError:
            print("bad packet")


            
#Set up reading and parsing threads
writeQueue = queue.Queue()
parseQueue = queue.Queue()
stopEvent = threading.Event()

def storeAndParse(data):
    writeQueue.put(data)
    parseQueue.put(data)
    

dataSerial = serial.Serial("COM11", 921600);

captureThread =threading.Thread(target = captureThreadMain, args=[dataSerial, storeAndParse])
#writeThread = threading.Thread(target = storeThreadMain, args=[writeQueue, stopEvent])
parseThread = threading.Thread(target = parseThreadMain, args=[parseQueue, stopEvent])

#Send UART commands to start the sensor
startSensor()

#Start processing threads
captureThread.start()
#writeThread.start()
parseThread.start()


