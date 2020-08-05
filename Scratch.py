import os
import sys
import cv2
import numpy as np

import matplotlib.pyplot as plt
from Triplet import Triplet
import sys

def main():
    temp = np.array(["this is a string", "This is another string"])
    tempFileName = (os.listdir('C:\\ATTFaceDataSet\\Training'))[0]
    tempFile = cv2.imread('C:\\ATTFaceDataSet\\Training\\{0}'.format(tempFileName))
    width = tempFile.shape[1]
    height = tempFile.shape[0]
    trainingImages = len(os.listdir("C:\\ATTFaceDataSet\\Training"))
    testImages = len(os.listdir("C:\\ATTFaceDataSet\\Testing"))
    train = np.empty((trainingImages, height, width),dtype=np.float)
    trainY = np.empty((trainingImages), dtype='<U23')  # This means string in numpy apparently
    test = np.empty((testImages, height, width),dtype=np.float)
    testY = np.empty((trainingImages), dtype='<U23')  # This means string in numpy apparently
    # load images
    i = 0
    for filename in os.listdir("C:\\ATTFaceDataSet\\Training"):
        y = filename.split('_')[0]
        trainY[i] = str(y)
        train[i] = cv2.imread('C:\\ATTFaceDataSet\\Training\\{0}'.format(filename), 0) / 255.0  # for color use 1
        i += 1

    j = 0
    for filename in os.listdir("C:\\ATTFaceDataSet\\Testing"):
        y = filename.split('_')[0]
        testY[j] = str(y)
        test[j] = cv2.imread('C:\\ATTFaceDataSet\\Testing\\{0}'.format(filename), 0) / 255.0
        j += 1

    trainX = train  # .reshape(train.shape[0],train.shape[1]*train.shape[2])
    testX = test  # .reshape(test.shape[0],test.shape[1]*test.shape[2])
