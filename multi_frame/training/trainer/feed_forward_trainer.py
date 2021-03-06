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


class FeedForwardTrainer(TrainerBase):
    def __init__(self, *params):
        super(FeedForwardTrainer, self).__init__(*params)

    def train(self, datasets):
        losses = []
        for dataset in datasets:
            # print("train: %s" % dataset)
            batch_losses = []
            for batch in dataset:
                xs, ts = batch
                with chainer.using_config('train', True):
                    loss = self.model.compute_loss(xs, ts)
                # 誤差逆伝播
                self.model.cleargrads()
                loss.backward()
                self.optimizer.update()
                batch_losses.append(loss.data)
                # バッチ単位で更新する。
            losses.append(sum(batch_losses)/len(batch_losses))
        return sum(losses)/len(losses)

    def validate(self, datasets):
        losses = []
        for dataset in datasets:
            batch_losses = []
            for batch in dataset:
                xs, ts = batch
                with chainer.using_config('train', False):
                    loss = self.model.compute_loss(xs, ts)
                batch_losses.append(loss.data)
            losses.append(sum(batch_losses)/len(batch_losses))
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

