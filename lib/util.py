import serial
import time
import numpy as np

def startSensor(commandport):
    with serial.Serial(commandport,115200, parity=serial.PARITY_NONE) as controlSerial:
#        with open("68xx_pplcount_ES2.cfg", 'r') as configfile:
        with open("people_detection_and_tracking_50m_3D.cfg", 'r') as configfile:
            for line in configfile:
                print(">> " + line, flush = True)
                controlSerial.write(line.encode('ascii'))
                print("<< " + controlSerial.readline().decode('ascii'), flush = True) #echo
                print("<< " + controlSerial.readline().decode('ascii'), flush = True) #"done"
                controlSerial.read(11) #prompt
                time.sleep(0.01)
            print("sensor started")


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
