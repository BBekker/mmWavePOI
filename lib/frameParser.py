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
    POINT_CLOUD_SIDE_INFO = 9
    
    
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
        'trackingProcessingTime' / Int32ul,
        'uartSendingTime' / Int32ul,
        'numTLVs' / Int16ul, 
        'checksum' / Int16ul,
    ),
    #Probe(this.header),
    "packets" / Struct(
             "type" / Int32ul,
             "len" / Int32ul,
                # Probe(this.type),
                # Probe(this.len),
             "data" / Switch(this.type,
                {
                Message.POINT_CLOUD:
                    "objects" / Struct(
                        "range" / Float32l,
                        "azimuth" / Float32l,
                        "elevation" / Float32l,
                        "doppler" / Float32l,
                    )[lambda ctx: int((ctx.len) / 16)],


                Message.POINT_CLOUD_SIDE_INFO:
                    "objects" / Struct(
                        "snr" / Int16sl,
                        "noise" / Int16sl,
                    )[lambda ctx: int((ctx.len) / 4)],

                Message.TARGET_LIST: 
                    "targets" / Struct(
                        "tid" / Int32ul,
                        "posx" / Float32l,
                        "posy" / Float32l,
                        "velX" / Float32l,
                        "velY" / Float32l,
                        "accX" / Float32l,
                        "accY"/ Float32l,
                        "posZ" / Float32l,
                        "velZ" / Float32l,
                        "accZ" / Float32l
                    )[lambda ctx: int((ctx.len) / (10*4))],
                 
                Message.TARGET_INDEX:
                    "indices" / Int8ul[this.len],
                    
                Message.NOISE_PROFILE: 
                    Array(rangeFFTSize, Int16ul),
                    
                Message.AZIMUT_STATIC_HEAT_MAP: 
                    Array(rangeFFTSize * antennas, Struct("Img" / Int16sl, "Re" / Int16sl)),
                    
                Message.RANGE_DOPPLER_HEAT_MAP: 
                    Array(rangeFFTSize * dopplerFFTSize, Int16ul),

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
        return np.array([[d['range'], d['azimuth'], d['doppler'], d['snr'], d['elevation']] for d in self.points])

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



def parseFrame(rawData):

    parsedFrame = None
    if(rawData == None):
        return None
    try:
        frame = frameParser.parse(rawData)
        #print(frame)
        parsedFrame = ParsedFrame(frame['header']['frameNumber'])
        targetList = getPacket(frame, Message.TARGET_LIST)
        targetIndices = getPacket(frame, Message.TARGET_INDEX)
        pointCloudLocation = getPacket(frame, Message.POINT_CLOUD)
        sideInfo = getPacket(frame, Message.POINT_CLOUD_SIDE_INFO)

        pointCloud = [{**pointCloudLocation[i], **sideInfo[i]} for i in range(len(pointCloudLocation))]

        # print(len(targetIndices))
        # print(len(pointCloudLocation), len(sideInfo))

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

    except StreamError:
        print("bad packet")

    return parsedFrame


if __name__ == "__main__":
    import sys
    file = sys.argv[1]
    parsed = GreedyRange(frameParser).parse_file(file)
    print(len(parsed))