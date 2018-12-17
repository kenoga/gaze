# -*- coding: utf-8 -*-
import os, sys
import json
import chainer
import pickle

sys.path.append(os.path.abspath(".."))
import train.model.cnn as cnn
from train.utils.DataLoader import ImageLoader

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer import cuda

class GazeDetector(object):
    def __init__(self, model, model_path, size=(32, 96), gpu=True):
        self.img_loader = ImageLoader(size)
        serializers.load_npz(model_path, model)
        self.gpu = gpu
        if self.gpu:
            model.to_gpu()
        self.model = model
        
    def _path2x(self, path):
        x = self.img_loader.load(path)
        x = np.array([x], dtype=x.dtype)
        if self.gpu:
            x = cuda.to_gpu(x)
        x = chainer.Variable(x)
        return x

    def detect(self, path):
        x = self._path2x(path)
        return F.softmax(self.model(x)).data.tolist()[0]
    
    def get_fc1(self, path):
        x = self._path2x(path)
        return self.model.get_fc1(x).data.tolist()[0]
    
    def get_fc2(self, path):
        x = self._path2x(path)
        return self.model.get_fc2(x).data.tolist()[0]
    
