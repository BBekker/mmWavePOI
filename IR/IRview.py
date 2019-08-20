import pyqtgraph as pg
import numpy as np
import serial
import pyqtgraph.multiprocess as mp

proc = mp.QtProcess(processRequests=False)
rpg = proc._import('pyqtgraph')
image = rpg.image(np.zeros((24,32)))
with serial.Serial('COM16') as serial:
    serial.readline(10000)
    while(True):
        line = serial.readline(10000).decode('utf-8')
        values = line.split(',')[:-1]
        values = list(map(float, values))

        image.setImage(np.reshape(np.array(values), (24,32)))
