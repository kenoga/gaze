# -*- coding: utf-8 -*-
import os
import json
import chainer
import numpy as np

import chainer.links as L
import chainer.functions as F


class CNN(chainer.Chain):
    def __init__(self, class_num):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(in_channels=None, out_channels=64, ksize=3, nobias=False)
            self.conv1_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=3, nobias=False)
            self.conv2_1 = L.Convolution2D(in_channels=None, out_channels=256, ksize=3, nobias=False)
            self.conv2_2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=3, nobias=False)
            self.fc1 = L.Linear(None, 256, nobias=False)
            self.fc2 = L.Linear(None, 128, nobias=False)
            self.fc3 = L.Linear(None, 2, nobias=False)      

    def __call__(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, ksize=3)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, ksize=3)
        h = F.dropout(F.relu(self.fc1(h)))
        h = F.dropout(F.relu(self.fc2(h)))
        y = self.fc3(h)
        return y
        


class CNNWithFCFeature(chainer.Chain):

    def __init__(self, class_num):
        super(CNNWithFCFeature, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 20, ksize=5, nobias=False)
            self.conv1_2 = L.Convolution2D(None, 50, ksize=5, nobias=False)
            # self.conv2_1 = L.Convolution2D(None, 256, ksize=3, nobias=False)
            # self.conv2_2 = L.Convolution2D(None, 128, ksize=3, nobias=False)
            self.fc1 = L.Linear(None, 256, nobias=False)
            self.fc2 = L.Linear(None, 128, nobias=False)
            self.fc3 = L.Linear(None, 2, nobias=False)      

    def __call__(self, x, feature):
        h = F.relu(self.conv1_1(x))
        h = F.max_pooling_2d(h, ksize=2)
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, ksize=2)
        h = F.dropout(F.relu(self.fc1(h)))
        # fcにfeatureを結合
        h = F.hstack((h, feature))
        h = F.dropout(F.relu(self.fc2(h)))
        y = self.fc3(h)
        return y
    
 
# class AlexLike1(chainer.Chain):

#     """Single-GPU AlexNet without partition toward the channel axis."""

# #     insize = 128

#     def __init__(self, n_out):
#         super(AlexLike1, self).__init__(
#             conv1=L. Convolution2D(3,  32, 8, stride=4),
#             conv2=L.Convolution2D(32, 256,  5, pad=2),
#             conv3=L. Convolution2D(256, 256,  3, pad=1),
#             conv4=L.Convolution2D(256, 256,  3, pad=1),
#             conv5=L. Convolution2D(256, 32,  3, pad=1),
#             fc6=L.Linear(None, 144),
#             fc7=L.Linear(None, 50),
#             fc8=L.Linear(None, n_out),
#         )
#         self.train = True

#     def __call__(self, x):
#         h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv1(x))), 3, stride=2)
#         h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv2(h))), 3, stride=2)
#         h = F.relu(self.conv3(h))
#         h = F.relu(self.conv4(h))
#         h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
# #         np.append(h , y)
#         h = F.dropout(F.relu(self.fc6(h)))
#         h = F.dropout(F.relu(self.fc7(h)))
#         h = self.fc8(h)
#         return h
