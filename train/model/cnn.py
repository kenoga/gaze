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


class SpatialWeightsCNN(chainer.Chain):
    def __init__(self):
        super(SpatialWeightsCNN, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(in_channels=None, out_channels=64, ksize=2, nobias=False)
            self.conv1_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=2, nobias=False)
            self.conv2_1 = L.Convolution2D(in_channels=None, out_channels=256, ksize=2, nobias=False)
            self.conv2_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=2, nobias=False)
            self.fc1 = L.Linear(None, 256, nobias=False)
            self.fc2 = L.Linear(None, 128, nobias=False)
            self.fc3 = L.Linear(None, 2, nobias=False)

            self.conv_sw_1 = L.Convolution2D(in_channels=128, out_channels=64, ksize=1, nobias=False)
            self.conv_sw_2 = L.Convolution2D(in_channels=None, out_channels=32, ksize=1, nobias=False)
            self.conv_sw_3 = L.Convolution2D(in_channels=None, out_channels=1, ksize=1, nobias=False)


    def __call__(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, ksize=3)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, ksize=3)

        map = F.relu(self.conv_sw_1(h))
        map = F.relu(self.conv_sw_2(map))
        map = F.relu(self.conv_sw_3(map))

        map = F.tile(map, (1,128,1,1))
        h = h * map

        h = F.dropout(F.relu(self.fc1(h)))
        # h = F.hstack((h, feature))
        h = F.dropout(F.relu(self.fc2(h)))
        y = self.fc3(h)
        return y


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

class CNNWithFCFeature2(chainer.Chain):
    def __init__(self):
        super(CNNWithFCFeature2, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(in_channels=None, out_channels=64, ksize=3, nobias=False)
            self.conv1_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=3, nobias=False)
            self.conv2_1 = L.Convolution2D(in_channels=None, out_channels=256, ksize=3, nobias=False)
            self.conv2_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=3, nobias=False)
            self.fc1 = L.Linear(None, 256, nobias=False)
            self.fc2 = L.Linear(None, 128, nobias=False)
            self.fc3 = L.Linear(None, 2, nobias=False)

            self.fc_for_feature = L.Linear(136, 16)


    def __call__(self, x, feature):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, ksize=3)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, ksize=3)
        h = F.dropout(F.relu(self.fc1(h)))

        feature = F.relu(self.fc_for_feature(feature))

        h = F.hstack((h, feature))
        h = F.dropout(F.relu(self.fc2(h)))
        y = self.fc3(h)
        return y

class CNNWithFCFeature3(chainer.Chain):
    def __init__(self):
        super(CNNWithFCFeature3, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(in_channels=None, out_channels=64, ksize=3, nobias=False)
            self.conv1_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=3, nobias=False)
            self.conv2_1 = L.Convolution2D(in_channels=None, out_channels=256, ksize=3, nobias=False)
            self.conv2_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=3, nobias=False)
            self.fc1 = L.Linear(None, 256, nobias=False)
            self.fc2 = L.Linear(None, 128, nobias=False)
            self.fc3 = L.Linear(None, 2, nobias=False)

            self.fc_for_feature1 = L.Linear(136, 64)
            self.fc_for_feature2 = L.Linear(64, 16)


    def __call__(self, x, feature):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, ksize=3)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, ksize=3)
        h = F.dropout(F.relu(self.fc1(h)))

        feature = F.relu(self.fc_for_feature1(feature))
        feature = F.relu(self.fc_for_feature2(feature))

        h = F.hstack((h, feature))
        h = F.dropout(F.relu(self.fc2(h)))
        y = self.fc3(h)
        return y


class CNNEachEye1(chainer.Chain):
    def __init__(self):
        super(CNNEachEye1, self).__init__()
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
        h = F.max_pooling_2d(h, ksize=3)
        eye1_h = F.dropout(F.relu(self.fc1(h)))

        h = F.relu(self.conv1_1(eye2))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, ksize=3)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, ksize=3)
        eye2_h = F.dropout(F.relu(self.fc1(h)))

        h = F.concat((eye1_h, eye2_h), axis=1)

        h = F.dropout(F.relu(self.fc2(h)))
        y = self.fc3(h)
        return y


class CNNEachEye2(chainer.Chain):
    def __init__(self):
        super(CNNEachEye2, self).__init__()
        with self.init_scope():
            self.conv1_1_1 = L.Convolution2D(in_channels=None, out_channels=32, ksize=3, nobias=False)
            self.conv1_1_2 = L.Convolution2D(in_channels=None, out_channels=64, ksize=3, nobias=False)
            self.conv1_2_1 = L.Convolution2D(in_channels=None, out_channels=256, ksize=3, nobias=False)
            self.conv1_2_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=3, nobias=False)

            self.conv2_1_1 = L.Convolution2D(in_channels=None, out_channels=64, ksize=3, nobias=False)
            self.conv2_1_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=3, nobias=False)
            self.conv2_2_1 = L.Convolution2D(in_channels=None, out_channels=256, ksize=3, nobias=False)
            self.conv2_2_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=3, nobias=False)
            self.fc1 = L.Linear(None, 256, nobias=False)
            self.fc2 = L.Linear(None, 128, nobias=False)
            self.fc3 = L.Linear(None, 2, nobias=False)

    def __call__(self, eye1, eye2):
        h = F.relu(self.conv1_1_1(eye1))
        h = F.relu(self.conv1_1_2(h))
        h = F.max_pooling_2d(h, ksize=3)
        h = F.relu(self.conv1_2_1(h))
        h = F.relu(self.conv1_2_2(h))
        h = F.max_pooling_2d(h, ksize=3)
        eye1_h = F.dropout(F.relu(self.fc1(h)))

        h = F.relu(self.conv2_1_1(eye2))
        h = F.relu(self.conv2_1_2(h))
        h = F.max_pooling_2d(h, ksize=3)
        h = F.relu(self.conv2_2_1(h))
        h = F.relu(self.conv2_2_2(h))
        h = F.max_pooling_2d(h, ksize=3)
        eye2_h = F.dropout(F.relu(self.fc1(h)))

        h = F.concat((eye1_h, eye2_h), axis=1)

        h = F.dropout(F.relu(self.fc2(h)))
        y = self.fc3(h)
        return y

class CNNEachEyeWithAttention1(chainer.Chain):

    def __init__(self):
        super(CNNEachEyeWithAttention1, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(in_channels=None, out_channels=64, ksize=3, nobias=False)
            self.conv1_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=3, nobias=False)
            self.conv2_1 = L.Convolution2D(in_channels=None, out_channels=256, ksize=3, nobias=False)
            self.conv2_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=3, nobias=False)
            self.fc1 = L.Linear(None, 256, nobias=False)
            self.fc2 = L.Linear(None, 128, nobias=False)
            self.fc3 = L.Linear(None, 2, nobias=False)

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


        h = F.concat((eye1_h, eye2_h), axis=1)

        y = self.fc3(h)
        return y



class CNNEachEyeWithAttention4(chainer.Chain):

    def __init__(self):
        super(CNNEachEyeWithAttention4, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(in_channels=None, out_channels=64, ksize=3, nobias=False)
            self.conv1_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=3, nobias=False)
            self.conv2_1 = L.Convolution2D(in_channels=None, out_channels=256, ksize=3, nobias=False)
            self.conv2_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=3, nobias=False)
            self.fc1 = L.Linear(None, 128, nobias=False)
            self.fc2 = L.Linear(None, 64, nobias=False)
            self.fc3 = L.Linear(None, 32, nobias=False)
            self.fc4 = L.Linear(None, 2, nobias=False)

            self.fc_for_attention = L.Linear(136, 128, nobias=False)

    def __call__(self, eye1, eye2, face):
        h = F.relu(self.conv1_1(eye1))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, ksize=3)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        eye1_h = F.max_pooling_2d(h, ksize=3)
        eye1_h = F.relu(self.fc1(eye1_h))
        eye1_h = F.relu(self.fc2(eye1_h))

        h = F.relu(self.conv1_1(eye2))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, ksize=3)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        eye2_h = F.max_pooling_2d(h, ksize=3)
        eye2_h = F.relu(self.fc1(eye2_h))
        eye2_h = F.relu(self.fc2(eye2_h))

        h = F.concat((eye1_h, eye2_h), axis=1)

        ats = F.sigmoid(self.fc_for_attention(face))
        h = h * ats

        h = F.dropout(F.relu(self.fc3(h)))
        y = self.fc4(h)
        return y


class CNNEachEyeWithAttention5(chainer.Chain):

    def __init__(self):
        super(CNNEachEyeWithAttention5, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(in_channels=None, out_channels=64, ksize=3, nobias=False)
            self.conv1_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=3, nobias=False)
            self.conv2_1 = L.Convolution2D(in_channels=None, out_channels=256, ksize=3, nobias=False)
            self.conv2_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=3, nobias=False)
            self.fc1 = L.Linear(None, 128, nobias=False)
            self.fc2 = L.Linear(None, 64, nobias=False)
            self.fc3 = L.Linear(None, 32, nobias=False)
            self.fc4 = L.Linear(None, 2, nobias=False)

            self.fc_for_attention1 = L.Linear(136, 256, nobias=False)
            self.fc_for_attention2 = L.Linear(256, 128, nobias=False)

    def __call__(self, eye1, eye2, face):
        h = F.relu(self.conv1_1(eye1))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, ksize=3)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        eye1_h = F.max_pooling_2d(h, ksize=3)
        eye1_h = F.relu(self.fc1(eye1_h))
        eye1_h = F.relu(self.fc2(eye1_h))

        h = F.relu(self.conv1_1(eye2))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, ksize=3)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        eye2_h = F.max_pooling_2d(h, ksize=3)
        eye2_h = F.relu(self.fc1(eye2_h))
        eye2_h = F.relu(self.fc2(eye2_h))

        h = F.concat((eye1_h, eye2_h), axis=1)

        ats = F.sigmoid(self.fc_for_attention1(face))
        ats = F.sigmoid(self.fc_for_attention2(face))
        h = h * ats

        h = F.dropout(F.relu(self.fc3(h)))
        y = self.fc4(h)
        return y
