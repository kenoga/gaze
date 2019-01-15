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

class TrainerBase(object):
    def __init__(self, conf):
        for key, value in conf.items():
            self.__dict__[key] = value
        self.set_random_seed(1107)

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        cp.random.seed(seed)

    def train(self, dataset):
        pass

    def validate(self, dataset):
        pass

    def test(self, dataset):
        pass


class NStepTrainer(TrainerBase):
    def __init__(self, config):
        super(NStepTrainer, self).__init__(config)

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        cp.random.seed(seed)

    def train(self, datasets):
        losses = []
        for dataset in datasets:
            print("train: %s" % dataset)
            batch_losses = []
            for i, batch in enumerate(dataset):
                xs, ts = batch
                with chainer.using_config('train', True):
                    loss = self.model.compute_loss(xs, ts)
                batch_losses.append(loss.data)
                # 誤差逆伝播
                self.model.cleargrads()
                loss.backward()
                # バッチ単位で古い記憶を削除し、計算コストを削減する。
                loss.unchain_backward()
                # バッチ単位で更新する。
                self.optimizer.update()
                 # AttetionLSTMのときはバッチごとに状態をリセット
                if self.network == AttentionLSTM:
                    self.model.reset_state()
            losses.append(sum(batch_losses)/len(batch_losses))
            self.model.reset_state()
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
            self.model.reset_state()
        return sum(losses)/len(losses)

    def test(self, dataset, all_result=False):
        ys_all = [None for i in range(dataset.batch_size)]
        ts_all = [None for i in range(dataset.batch_size)]
        losses = []
        for batch in dataset:
            xs, ts = batch
            with chainer.using_config('train', False):
                ys = self.model(xs)

            # バッチごとに分割されているデータを元の順番に戻す
            for batch_i in range(len(ys)):
                ys_all[batch_i] = ys[batch_i] if ys_all[batch_i] is None else F.vstack((ys_all[batch_i], ys[batch_i]))
                ts_all[batch_i] = ts[batch_i] if ts_all[batch_i] is None else F.hstack((ts_all[batch_i], ts[batch_i]))

        ys_all = F.concat(ys_all, axis=0)
        ts_all = F.concat(ts_all, axis=0)
        f1_score = F.f1_score(ys_all, ts_all)[0][1].data
        if all_result:
            return f1_score, (ts_all, ys_all)
        return f1_score

    def _setup(self):
        if self.network == AttentionLSTM or self.network == AttentionGRU:
            self.model = self.network(self.rnn_layer, self.rnn_input, self.rnn_hidden, self.rnn_output, self.window_size, self.dropout)
        else:
            self.model = self.network(self.rnn_layer, self.rnn_input, self.rnn_hidden, self.rnn_output, self.dropout)
        if self.gpu >= 0:
            chainer.cuda.get_device(self.gpu).use()
            self.model.to_gpu()
        # Optimizer
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

    def get_exp_id(self, val_dialog_id):
        # network_inputType_rnnHidden_batchSize_windowSize_trainSize_valDialogId
        return "%s_%s_%04d_%02d_%02d_%02d_%02d" % \
        (self.network.name, self.input_type, self.rnn_hidden, self.batch_size, self.window_size, len(self.train_ids)-1, val_dialog_id)
