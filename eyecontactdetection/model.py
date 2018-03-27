import os
import json
import chainer

import chainer.links as L
import chainer.functions as F


class CNN(chainer.Chain):

    def __init__(self, class_num):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 16, ksize=5, pad=2, nobias=True)
            self.conv1_2 = L.Convolution2D(None, 16, ksize=5, pad=2, nobias=True)
            self.conv2_1 = L.Convolution2D(None, 32, ksize=3, pad=1, nobias=True)
            self.conv2_2 = L.Convolution2D(None, 32, ksize=3, pad=1, nobias=True)
            self.fc1 = L.Linear(None, 512, nobias=True)
            self.fc2 = L.Linear(None, class_num, nobias=True)      

    def __call__(self, x):
        conv1_1 = self.conv1_1(x)
        conv1_1 = F.relu(conv1_1)
        conv1_2 = self.conv1_2(conv1_1)
        conv1_2 = F.relu(conv1_2)
        pool1 = F.max_pooling_2d(conv1_2, ksize=2, stride=2)
        conv2_1 = self.conv2_1(pool1)
        conv2_1 = F.relu(conv2_1)
        conv2_2 = self.conv2_2(conv2_1)
        conv2_2 = F.relu(conv2_2)
        pool2 = F.max_pooling_2d(conv2_2, ksize=2, stride=2)
        fc1 = self.fc1(pool2)
        fc1 = F.relu(fc1)
        fc2 = self.fc2(fc1)
        return fc2