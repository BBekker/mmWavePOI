from enum import Enum
from construct import *
import numpy as np
from datetime import datetime

rangeFFTSize = 256
dopplerFFTSize = 256
antennas = 8

class Message(Enum):
    DETECTED_POINTS = 1
    RANGE_PROFILE = 2 
    NOISE_PROFILE = 3
    AZIMUT_STATIC_HEAT_MAP = 4
    RANGE_DOPPLER_HEAT_MAP = 5
    POINT_CLOUD = 6
    TARGET_LIST = 7
    TARGET_INDEX = 8
    STATS = 9
    HEATMAP = 10
    MAX = 11
    
    
frameParser = Aligned(4,
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
        'uartSendingTime' / Int32ul,
        'trackingProcessingTime' / Int32ul,
        'numTLVs' / Int16ul, 
        'checksum' / Int16ul,
    ),
    "packets" / Struct(
             "type" / Int32ul,
             "len" / Int32ul,
             "data" / Switch(this.type,
                {
                Message.POINT_CLOUD: 
                    "objects" / Struct(
                        "range" / Float32l,
                        "angle" / Float32l,
                        "doppler" / Float32l,
                        "snr" / Float32l,
                    )[lambda ctx: int((ctx.len - 8) / 16)],
                
                Message.TARGET_LIST: 
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
                 
                Message.TARGET_INDEX:
                    "indices" / Int8ul[this.len - 8],
                    
                Message.NOISE_PROFILE: 
                    Array(rangeFFTSize, Int16ul),
                    
                Message.AZIMUT_STATIC_HEAT_MAP: 
                    Array(rangeFFTSize * antennas, Struct("Img" / Int16sl, "Re" / Int16sl)),
                    
                Message.RANGE_DOPPLER_HEAT_MAP: 
                    Array(rangeFFTSize * dopplerFFTSize, Int16ul),
                    
                Message.STATS: 
                    Struct(
                    "interFrameProcessingTime" / Int32ul,
                    "transmitOutputTime" / Int32ul,
                    "interFrameProcessingMargin" / Int32ul,
                    "interChirpProcessingMargin" / Int32ul,
                    "activeFrameCPULoad" / Int32ul,
                    "interFrameCPULoad" / Int32ul
                ),

                Message.HEATMAP:
                    Float32l[lambda ctx: int((ctx.len - 8) / 4)],
                }
                , default=Array(this.len, Byte))
         )[this.header.numTLVs] 
    )
)


def getPacket(frame, name):
    for packet in frame['packets']:
        if name == packet['type']:
            return packet['data']
    return []


def matchArrays(a, b):
    if len(a) != len(b):
        return False

    for i in range(0, len(a)):
        if a[i] != b[i]:
            return False
    return True


class Cluster:

    def __init__(self, tid):
        self.tid = tid
        self.points = []
        self.info = None

    def addPoint(self, point):
        self.points.append(point)

    def addInfo(self, info):
        self.info = info

    def getPoints(self):
        return np.array([[d['range'], d['angle'], d['doppler'], d['snr']] for d in self.points])

    def toDict(self):
        return {"tid": self.tid,  "class": 0, "points": self.getPoints()}


class ParsedFrame:

    def __init__(self, fn):
        self.clusters = []
        self.frameNumber = fn
        self.points = [] #Unclusterd points
        self.timestamp = datetime.now()

    def newCluster(self, tid):
        c = Cluster(tid)
        self.clusters.append(c)
        return c

    def addPoint(self, point):
        self.points.append(point)

    def toDict(self):
        return {"frameNumber": self.frameNumber,
                "timestamp": self.timestamp,
                "clusters": [c.toDict() for c in self.clusters]}



previousFrame = None
def parseFrame(rawData):
    global previousFrame

    parsedFrame = None
    if(rawData == None):
        return None
    try:
        frame = frameParser.parse(rawData)
        if previousFrame != None and previousFrame['header']['frameNumber'] == (frame['header']['frameNumber']-1):
            parsedFrame = ParsedFrame(frame['header']['frameNumber'])
            targetList = getPacket(frame, Message.TARGET_LIST)
            targetIndices = getPacket(frame, Message.TARGET_INDEX)
            pointCloud = getPacket(previousFrame, Message.POINT_CLOUD)

            # print(len(targetIndices), targetIndices)
            # print(len(pointCloud))

            #print(targetList, flush=True)
            for target in targetList:
                tid =  target['tid']
                cluster = parsedFrame.newCluster(tid)
                cluster.addInfo(target)
                for targetIndex in range(len(targetIndices)):
                    if targetIndices[targetIndex] == tid:
                        cluster.addPoint(pointCloud[targetIndex])

            #Add points without a cluster
            for targetIndex in range(len(targetIndices)):
                if targetIndices[targetIndex] >= 250:
                    parsedFrame.addPoint(pointCloud[targetIndex])       
        else:
            print("Frame skipped!", flush=True)
        previousFrame = frame
    except StreamError:
        print("bad packet")
        previousFrame = None

    return parsedFrame