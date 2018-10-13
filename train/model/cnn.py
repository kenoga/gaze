# -*- coding: utf-8 -*-
import os
import json
import chainer
import numpy as np

import chainer.links as L
import chainer.functions as F


class CNN(chainer.Chain):
    def __init__(self):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(in_channels=None, out_channels=64, ksize=3, nobias=False)
            self.conv1_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=3, nobias=False)
            self.conv2_1 = L.Convolution2D(in_channels=None, out_channels=256, ksize=3, nobias=False)
            self.conv2_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=3, nobias=False)
            self.fc1 = L.Linear(None, 256, nobias=False)
            self.fc2 = L.Linear(None, 128, nobias=False)
            self.fc3 = L.Linear(None, 2, nobias=False)
        self.init_flag = True

    def __call__(self, x):

        if self.init_flag:
            self._pshape(x)

        h = F.relu(self.conv1_1(x))
        if self.init_flag:
            self._pshape(h, "conv1_1")

        h = F.relu(self.conv1_2(h))
        if self.init_flag:
            self._pshape(h, "conv1_2")

        h = F.max_pooling_2d(h, ksize=3)
        if self.init_flag:
            self._pshape(h, "mp_1")

        h = F.relu(self.conv2_1(h))
        if self.init_flag:
            self._pshape(h, "conv2_1")

        h = F.relu(self.conv2_2(h))
        if self.init_flag:
            self._pshape(h, "conv2_2")

        h = F.max_pooling_2d(h, ksize=3)
        if self.init_flag:
            self._pshape(h, "conv2_2")

        h = F.dropout(F.relu(self.fc1(h)))
        if self.init_flag:
            self._pshape(h, "fc1")

        h = F.dropout(F.relu(self.fc2(h)))
        if self.init_flag:
            self._pshape(h, "fc2")

        y = self.fc3(h)
        if self.init_flag:
            self._pshape(h, "fc3")

        if self.init_flag:
            self.init_flag = False
        return y

    def _pshape(self, layer, label=""):
        print("-> (%s)" % label)
        print(layer.shape)

class CNNWithFCFeature(chainer.Chain):

    def __init__(self):
        super(CNNWithFCFeature, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(in_channels=None, out_channels=64, ksize=3, nobias=False)
            self.conv1_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=3, nobias=False)
            self.conv2_1 = L.Convolution2D(in_channels=None, out_channels=256, ksize=3, nobias=False)
            self.conv2_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=3, nobias=False)
            self.fc1 = L.Linear(None, 256, nobias=False)
            self.fc2 = L.Linear(None, 128, nobias=False)
            self.fc3 = L.Linear(None, 2, nobias=False)

#             self.conv1_1 = L.Convolution2D(None, 20, ksize=5, nobias=False)
#             self.conv1_2 = L.Convolution2D(None, 50, ksize=5, nobias=False)
#             # self.conv2_1 = L.Convolution2D(None, 256, ksize=3, nobias=False)
#             # self.conv2_2 = L.Convolution2D(None, 128, ksize=3, nobias=False)
#             self.fc1 = L.Linear(None, 256, nobias=False)
#             self.fc2 = L.Linear(None, 128, nobias=False)
#             self.fc3 = L.Linear(None, 2, nobias=False)

    def __call__(self, x, feature):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, ksize=3)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, ksize=3)
        h = F.dropout(F.relu(self.fc1(h)))
        h = F.hstack((h, feature))
        h = F.dropout(F.relu(self.fc2(h)))
        y = self.fc3(h)
        return y

class CNNEachEye(chainer.Chain):
    def __init__(self):
        super(CNNEachEye, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(in_channels=None, out_channels=64, ksize=3, nobias=False)
            self.conv1_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=3, nobias=False)
            self.conv2_1 = L.Convolution2D(in_channels=None, out_channels=256, ksize=3, nobias=False)
            self.conv2_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=3, nobias=False)
            self.fc1 = L.Linear(None, 256, nobias=False)
            self.fc2 = L.Linear(None, 128, nobias=False)
            self.fc3 = L.Linear(None, 2, nobias=False)

    def __call__(self, eye1, eye2):
        h = F.relu(self.conv1_1(eye1))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, ksize=3)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        eye1_h = F.max_pooling_2d(h, ksize=3)
        eye1_h = F.dropout(F.relu(self.fc1(h)))

        h = F.relu(self.conv1_1(eye2))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, ksize=3)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        eye2_h = F.max_pooling_2d(h, ksize=3)
        eye2_h = F.dropout(F.relu(self.fc1(h)))

        h = F.concat((eye1_h, eye2_h), axis=1)

        h = F.dropout(F.relu(self.fc2(h)))
        y = self.fc3(h)
        return y


class CNNEachEyeWithAttention(chainer.Chain):

    def __init__(self):
        super(CNNEachEyeWithAttention, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(in_channels=None, out_channels=64, ksize=3, nobias=False)
            self.conv1_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=3, nobias=False)
            self.conv2_1 = L.Convolution2D(in_channels=None, out_channels=256, ksize=3, nobias=False)
            self.conv2_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=3, nobias=False)
            self.fc1 = L.Linear(None, 256, nobias=False)
            self.fc2 = L.Linear(None, 128, nobias=False)
            self.fc3 = L.Linear(None, 2, nobians=False)

            self.fc_for_attention = L.Linear(136, 128, nobias=False)

    def __call__(self, eye1, eye2, face):
        h = F.relu(self.conv1_1(eye1))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, ksize=3)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        eye1_h = F.max_pooling_2d(h, ksize=3)
        eye1_h = F.dropout(F.relu(self.fc1(eye1_h)))
        eye1_h = F.dropout(F.relu(self.fc2(eye1_h)))

        h = F.relu(self.conv1_1(eye2))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, ksize=3)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        eye2_h = F.max_pooling_2d(h, ksize=3)
        eye2_h = F.dropout(F.relu(self.fc1(eye2_h)))
        eye2_h = F.dropout(F.relu(self.fc2(eye2_h)))

        ats = F.sigmoid(self.fc_for_attention(face))
        eye1_h = eye1_h * ats
        eye2_h = eye2_h * ats

        h = F.concat(eye1_h, eye2_h)

        y = self.fc3(h)
        return y
