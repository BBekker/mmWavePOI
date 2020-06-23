# mmWavePOI

Track, cluster and classify people using the ti mmwave sensor.

This repo is a messy collection of the stuff i did in my thesis. The usefull parts are:

QTGui.py is a capture program for mmwave data.
use `-h` to see the options

labeler/labeler.py is a program to inspect the recorded data, and add labels to them.

RFClassifier.py is a script to try different classification methods based on scikit-learn.

recurrent_networks.ipynb is a jupyter notebook using tensorflow to classify the tracked objects. Implemented is a LSTM and GRU network.

Spread throughout are scripts for inspecting, plotting, and various tools to work with the mmwave data and intermediate formats.
