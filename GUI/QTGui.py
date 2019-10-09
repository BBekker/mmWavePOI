import sys
import random
from PySide2 import QtCore, QtWidgets, QtGui,QtOpenGL
import pyqtgraph as pg
import pyqtgraph.opengl as pgl
import numpy as np
from worker import Worker
import time

class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.threadpool = QtCore.QThreadPool()

        self.hello = ["Hallo Welt", "Hei maailma", "Hola Mundo", "Привет мир"]

        self.button = QtWidgets.QPushButton("Click me!")
        self.text = QtWidgets.QLabel("Hello World")
        self.text.setAlignment(QtCore.Qt.AlignCenter)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)

        self.button.clicked.connect(self.magicExecutor)

        self.plotwindow = pg.PlotWidget()
        self.plotwindow.plot(np.random.rand((100)))

        def mouseMoved(event):
            pos = event
            self.text.setText(f"x:{pos.x()}, y:{pos.y()}")

        self.plotwindow.scene().sigMouseMoved.connect(mouseMoved)
        self.layout.addWidget(self.plotwindow)

        self.glview = pgl.GLViewWidget()
        grid = pgl.GLGridItem()
        scatterplot = pgl.GLScatterPlotItem()
        scatter2 = pgl.GLScatterPlotItem(color=[1.0, 0, 0, 0.2], size=50)
        # Draw the area we are viewing.
        background = pgl.GLLinePlotItem(pos=np.array([[-5, 0, 0], [5, 0, 0], [10, 25, 0], [-10, 25, 0], [-5, 0, 0]]),
                                       color=(1, 1, 1, 1), width=2, antialias=True, mode='line_strip')

        self.layout.addWidget(self.glview)
        self.glview.addItem(grid)
        self.glview.addItem(background)
        self.glview.addItem(scatterplot)
        self.glview.addItem(scatter2)



    def magicExecutor(self):
        self.threadpool.start(Worker(self.magic))

    def magic(self):
        time.sleep(1.0)
        self.text.setText(random.choice(self.hello))

if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = MyWidget()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec_())