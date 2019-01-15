# -*- coding: utf-8 -*-

import os, sys
sys.path.append(os.path.abspath(".."))
import random
import datetime
import pickle
import copy
import numpy as np
import cupy as cp
import chainer
import chainer.cuda
import chainer.functions as F
from chainer import optimizers, serializers
from trainer import TrainerBase
from network.feedforward import *


class SingleFrameTrainer(TrainerBase):
    def __init__(self, config):
        super(SingleFrameTrainer, self).__init__(config)

    def train(self, dataset):
        losses = []
        for batch in dataset:
            xs, ts = batch
            with chainer.using_config('train', True):
                loss = self.model.compute_loss(xs, ts)
            # 誤差逆伝播
            self.model.cleargrads()
            loss.backward()
            self.optimizer.update()
            losses.append(loss.data)
            # バッチ単位で更新する。
        return sum(losses)/len(losses)

    def validate(self, dataset):
        losses = []
        for batch in dataset:
            xs, ts = batch
            with chainer.using_config('train', False):
                loss = self.model.compute_loss(xs, ts)
            losses.append(loss.data)
        return sum(losses)/len(losses)

    def test(self, dataset, all_result=False):
        ys_all = []
        ts_all = []
        for batch in dataset:
            xs, ts = batch
            with chainer.using_config('train', False):
                ys = self.model(xs)
            ys_all.append(ys)
            ts_all.append(ts)
        ys_all = F.concat(ys_all, axis=0)
        ts_all = F.concat(ts_all, axis=0)
        f1_score = F.f1_score(ys_all, ts_all)[0][1].data
        if all_result:
            return f1_score, (ts_all, ys_all)
        return f1_score

    def _setup(self):
        self.model = self.network(self.nn_input, self.rnn_output)
        if self.gpu >= 0:
            chainer.cuda.get_device(self.gpu).use()
            self.model.to_gpu()
        # Optimizer
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

    def get_exp_id(self, val_dialog_id):
        if self.network == OneLayerFeedForwardNeuralNetwork:
            network = "ff1"
        elif self.network == TwoLayerFeedForwardNeuralNetwork:
            network = "ff2"
        else:
            network = "unknown"
        # network_inputType_batchSize_trainSize_valDialogId
        return "%s_%s_%04d_%02d_%02d" % \
        (network, self.input_type, self.batch_size, len(self.train_ids)-1, val_dialog_id)
