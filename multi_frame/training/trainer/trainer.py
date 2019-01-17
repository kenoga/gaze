
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
from chainer import optimizers, serializers

class TrainerBase(object):
    def __init__(self, network, network_params, gpu=0):
        self.network = network
        self.network_params = network_params
        self.gpu = 0
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
    
    def _setup(self):
        self.model = self.network(**self.network_params)
        if self.gpu >= 0:
            chainer.cuda.get_device(self.gpu).use()
            self.model.to_gpu()
        # Optimizer
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

class CrossValidationTrainerWrapper(object):
    def __init__(self, trainer, dataset_iterator, epoch, log_path, npz_dir, loss_dir, config_id):
        self.trainer = trainer
        self.dataset_iterator = dataset_iterator
        self.epoch = epoch
        self.log_path = log_path
        self.npz_dir = npz_dir
        self.loss_dir = loss_dir
        self.config_id = config_id

    def cross_validate(self):
        self.trainer._setup()
        for train_datasets, val_datasets in self.dataset_iterator:
            val_dialog_id = self.dataset_iterator.current_val_dialog_id
            # 実験idを取得
            exp_id = "_".join([self.config_id, "%02d" % val_dialog_id])
            print(exp_id)

            # Set up a model and a optimizer
            self.trainer._setup()

            # Training
            train_losses = []
            val_losses = []
            min_val_loss = None
            min_val_model = None
            for epoch in range(self.epoch):
                train_loss = self.trainer.train(train_datasets)
                val_loss = self.trainer.validate(val_datasets)
                print('epoch:{}, loss:{}, val_loss:{}'.format(epoch, train_loss, val_loss))

                if min_val_loss is None or val_loss <= min_val_loss:
                    min_val_model = copy.deepcopy(self.trainer.model)
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

            f1_scores = [self.trainer.test(val_dataset) for val_dataset in val_datasets]
            ave_score = sum(f1_scores) / len(f1_scores)

            with open(self.log_path, "a") as fr:
                report = ", ".join(["VALIDATION_REPORT", exp_id, "%.4f" % min_val_loss, "%.4f" % ave_score])
                print(report)
                fr.write(report + "\n")
