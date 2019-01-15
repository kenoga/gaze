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
from single_frame_dataset_loader import SingleFrameDatasetsIteratorForCrossValidation
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


class SingleFrameCrossValidationTrainer(SingleFrameTrainer):
    def __init__(self, config):
        super(SingleFrameCrossValidationTrainer, self).__init__(config)
 
    def cross_validate(self):
        '''
        test_dialog_id: テスト用であるために、学習, 検証から外したいdialog_id
        '''
        cross_validation_dataset_loader = SingleFrameDatasetsIteratorForCrossValidation(self.dataset_path, self.batch_size, xp=self.xp, test_ids=self.test_ids, train_ids=self.train_ids, iterator=self.data_iterator)

        for train_dataset, val_dataset in cross_validation_dataset_loader:
            val_dialog_id = cross_validation_dataset_loader.current_val_dialog_id
            exp_id = self.get_exp_id(val_dialog_id)
            print(exp_id)
             
            # Set up a model and a optimizer
            self._setup()

            # Training
            train_losses = []
            val_losses = []
            min_val_loss = None
            min_val_model = None
            for epoch in range(self.epoch):
                train_loss = self.train(train_dataset)
                val_loss = self.validate(val_dataset)
                print('epoch:{}, loss:{}, val_loss:{}'.format(epoch, train_loss, val_loss))

                if min_val_loss is None or val_loss <= min_val_loss:
                    min_val_model = copy.deepcopy(self.model)
                    min_val_loss = val_loss

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                with open(self.log_path, "a") as fr:
                    report = ", ".join(["EPOCH_REPORT", exp_id, "%.4f" % val_loss])
                    print(report)
                    fr.write(report + "\n")

            # 全エポックでもっともval_lossが低くなったモデルのfスコアを計算してモデルを保存
            npz_path = os.path.join(self.npz_dir, "%s.npz" % exp_id)
            serializers.save_npz(npz_path, min_val_model) 

            ave_score = self.test(val_dataset)
            with open(self.log_path, "a") as fr:
                report = ", ".join(["VALIDATION_REPORT", exp_id, "%.4f" % min_val_loss, "%.4f" % ave_score])
                print(report)
                fr.write(report + "\n")

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
        