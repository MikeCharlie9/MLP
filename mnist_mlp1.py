# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 14:49:10 2019

@author: machang
"""
import numpy as np
import struct
from numpy import random
import time
# import matplotlib.pyplot as plt

trainImages = "./mnist/train-images.idx3-ubyte"
trainLabels = "./mnist/train-labels.idx1-ubyte"
testImages = "./mnist/t10k-images.idx3-ubyte"
testLabels = "./mnist/t10k-labels.idx1-ubyte"

L1 = 784
L2 = 100
L3 = 10

train_count = 60000  # 60000
test_count = 10000

learningRate = 0.3

weight1 = random.uniform(-0.1, 0.1, (L1, L2))
bias1 = np.matrix(np.zeros(L2))
weight2 = random.uniform(-0.1, 0.1, (L2, L3))
bias2 = np.matrix(np.zeros(L3))




def sigmoid(data):
    return 1/(1+np.exp(-data))


def back_sig(data):
    return np.multiply(data, (1 - data))


def readFile(filename):
    with open(filename, 'rb') as f:
        buf = f.read()
    return buf


def getImage(filename):
    buf = readFile(filename)
    image_index = 0
    image_index += struct.calcsize('>IIII')
    magic, numImages, imgRows, imgCols = struct.unpack_from('>IIII', buf, 0)

    images = np.zeros([numImages, L1], int)

    for i in range(numImages):
        temp = struct.unpack_from('>784B', buf, image_index)
        images[i] = np.array(temp)
        for j in range(784):
            if images[i][j] > 127:
                images[i][j] = 1
            else:
                images[i][j] = 0
        image_index += struct.calcsize('>784B')
    # print(magic,numImages,imgRows,imgCols)
    return images


def getLabel(filename):
    buf = readFile(filename)
    label_index = 0
    label_index += struct.calcsize('>II')
    magic, numLabels = struct.unpack_from('>II', buf, 0)

    labels = np.zeros([numLabels], int)

    for i in range(numLabels):
        temp = struct.unpack_from('>B', buf, label_index)
        labels[i] = np.array(temp)
        label_index += struct.calcsize('>B')
#    print(magic,numLabels)
    return labels


def findMax(data):
    maxNum = 0.0
    maxPos = 0
    i = 0
    for element in data:
        if element.astype('float32') > maxNum:
            maxNum = element.astype('float32')
            maxPos = i
        i += 1
    return maxPos


def train(images,labels):
    
    global weight1
    global weight2
    global bias1
    global bias2
    
#    images = getImage(trainImages)
#    labels = getLabel(trainLabels)
    standard = np.matrix(np.zeros([10]))
    delta1 = np.matrix(np.zeros([L2]))
    delta2 = np.matrix(np.zeros([L3]))
#    print("start train")
    for i in range(train_count):

        layer1 = sigmoid(np.dot(np.matrix([images[i]]), weight1)+bias1)
        layer2 = sigmoid(np.dot(layer1, weight2)+bias2)
#        print(i)

        standard[:, labels[i]] = 1
        delta2 = np.multiply((layer2-standard), back_sig(layer2))
        bias2 += -learningRate*delta2
        weight2 += -learningRate*(np.dot(layer1.T, delta2))

        delta1 = np.multiply(np.dot(delta2, weight2.T), back_sig(layer1))
        bias1 += -learningRate*delta1
        weight1 += -learningRate * (np.dot(np.matrix([images[i]]).T, delta1))

        standard[:, labels[i]] = 0
#    print("train over")
    


def inference(images,labels):
#    images = getImage(testImages)
#    labels = getLabel(testLabels)
    count = 0
    for i in range(test_count):

        layer1 = sigmoid(np.dot(np.matrix([images[i]]), weight1)+bias1)
        layer2 = sigmoid(np.dot(layer1, weight2)+bias2)
        maxPos = findMax(np.array(layer2)[0, :])
        if maxPos == labels[i]:
            count += 1
    return count


if __name__ == "__main__":
    
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    trainImagesData=getImage(trainImages)
    trainLabelsData=getLabel(trainLabels)
    testImagesData=getImage(testImages)
    testLabelsData=getLabel(testLabels)    
    print("start train")
    for epoch in range(10):
        print("epoch:",epoch,end="")
        train(trainImagesData,trainLabelsData)
        print(",correct:",inference(testImagesData,testLabelsData),"/10000")
    print("end")
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
