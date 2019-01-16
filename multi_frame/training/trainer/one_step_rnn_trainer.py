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

class OneStepRNNTrainer(TrainerBase):
    def __init__(self, config):
        super(OneStepRNNTrainer, self).__init__(config)

    def train(self, datasets):
        # 実装の都合上[dataset]が帰ってくるので0番目のみ使う
        dataset = datasets[0]
        loss = 0
        count = 0
        for xs, ts in dataset:
            count += 1
            with chainer.using_config('train', True):
                loss += self.model.compute_loss(xs, ts)
            # BPTTの幅はwindow_sizeとする
            if count % self.window_size == 0:
                self.model.cleargrads()
                loss.backward()
                self.optimizer.update()             
                loss.unchain_backward()
        self.model.reset_state()
        return loss.data

    def validate(self, datasets):
        dataset = datasets[0]
        loss = 0
        count = 0
        for xs, ts in dataset:
            with chainer.using_config('train', False):
                loss += self.model.compute_loss(xs, ts)
        self.model.reset_state()
        return loss.data


    def test(self, dataset, all_result=False):
        batch_size = dataset.batch_size
        ys_all = [[] for i in range(batch_size)]
        ts_all = [[] for i in range(batch_size)]
        for batch in dataset:
            xs, ts = batch
            with chainer.using_config('train', False):
                ys = self.model(xs)
            
            # バッチごとに分割されているデータを元の順番に戻す
            for batch_i in range(len(ys)):
                ys_all[batch_i].append(F.reshape(ys[batch_i], (1, ys[batch_i].shape[0])))
                ts_all[batch_i].append(F.reshape(ts[batch_i], (1,)))
#                 ys_all[batch_i] = ys[batch_i] if ys_all[batch_i] is None else F.vstack((ys_all[batch_i], ys[batch_i]))
#                 ts_all[batch_i] = ts[batch_i] if ts_all[batch_i] is None else F.hstack((ts_all[batch_i], ts[batch_i]))
        for i in range(batch_size):
            ys_all[i] = F.concat(ys_all[i], axis=0)
            ts_all[i] = F.concat(ts_all[i], axis=0)
        
        ys_all = F.concat(ys_all, axis=0)
        ts_all = F.concat(ts_all, axis=0)
        f1_score = F.f1_score(ys_all, ts_all)[0][1].data
        if all_result:
            return f1_score, (ts_all, ys_all)
        return f1_score
    

    def _setup(self):
        self.model = self.network(self.rnn_input, self.rnn_hidden, self.rnn_output, self.window_size)
        if self.gpu >= 0:
            chainer.cuda.get_device(self.gpu).use()
            self.model.to_gpu()
        # Optimizer
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        
    def get_exp_id(self, val_dialog_id):
        # network_inputType_rnnHidden_splitNum_windowSize_trainSize_valDialogId
          return "%s_%s_%04d_%02d_%02d_%02d_%02d" % \
        (self.network.name, self.input_type, self.rnn_hidden, self.split_num, self.window_size, len(self.train_ids)-1, val_dialog_id)
