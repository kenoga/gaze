# -*- coding: utf-8 -*-
import os
import json
import chainer

import chainer.links as L
import chainer.functions as F


class CNN(chainer.Chain):
    def __init__(self, class_num):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 64, ksize=3, nobias=False)
            self.conv1_2 = L.Convolution2D(None, 128, ksize=3, nobias=False)
            self.conv2_1 = L.Convolution2D(None, 256, ksize=3, nobias=False)
            self.conv2_2 = L.Convolution2D(None, 128, ksize=3, nobias=False)
            self.fc1 = L.Linear(None, 128, nobias=True)
            self.fc2 = L.Linear(None, 2, nobias=True)      

    def __call__(self, x):
        h = self.conv1_1(x)
        h = F.relu(h)
        h = self.conv1_2(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, ksize=3)
#         h = self.conv2_1(h)
#         h = F.relu(h)
#         h = self.conv2_2(h)
#         h = F.relu(h)
#         h = F.max_pooling_2d(h, ksize=3)
        h = F.dropout(F.relu(self.fc1(h)))
        y = self.fc2(h)
        return y

class CNNWithFCFeature(chainer.Chain):

    def __init__(self, class_num):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 64, ksize=3, nobias=False)
            self.conv1_2 = L.Convolution2D(None, 128, ksize=3, nobias=False)
            self.conv2_1 = L.Convolution2D(None, 256, ksize=3, nobias=False)
            self.conv2_2 = L.Convolution2D(None, 128, ksize=3, nobias=False)
            self.fc1 = L.Linear(None, 128, nobias=True)
            self.fc2 = L.Linear(None, 2, nobias=True)      

    def __call__(self, x, feature):
        h = self.conv1_1(x)
        h = F.relu(h)
        h = self.conv1_2(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, ksize=3)
#         h = self.conv2_1(h)
#         h = F.relu(h)
#         h = self.conv2_2(h)
#         h = F.relu(h)
#         h = F.max_pooling_2d(h, ksize=3)
        # fcにfeatureを結合        
        h = np.hstack((h, feature))
        h = F.dropout(F.relu(self.fc1(h)))
        y = self.fc2(h)
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