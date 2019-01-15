
# -*- coding: utf-8 -*-

import os, sys
sys.path.append(os.path.abspath(".."))
import random
import datetime
import pickle
import copy
import numpy as np
import cupy as cp

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

class CrossValidationTrainerWrapper(object):
    def __init__(self, trainer, dataset_iterator):
        self.trainer = trainer
        self.dataset_iterator = dataset_iterator
        # super(SingleFrameCrossValidationTrainer, self).__init__(config)

    def cross_validate(self):
        '''
        test_dialog_id: テスト用であるために、学習, 検証から外したいdialog_id
        '''

        for train_dataset, val_dataset in self.dataset_iterator:
            val_dialog_id = dataset_iterator.current_val_dialog_id
            # 実験idを取得
            exp_id = self.trainer.get_exp_id(val_dialog_id)
            print(exp_id)

            # Set up a model and a optimizer
            self.trainer._setup()

            # Training
            train_losses = []
            val_losses = []
            min_val_loss = None
            min_val_model = None
            for epoch in range(self.trainer.epoch):
                train_loss = self.trainer.train(train_dataset)
                val_loss = self.trainer.validate(val_dataset)
                print('epoch:{}, loss:{}, val_loss:{}'.format(epoch, train_loss, val_loss))

                if min_val_loss is None or val_loss <= min_val_loss:
                    min_val_model = copy.deepcopy(self.trainer.model)
                    min_val_loss = val_loss

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                with open(self.trainer.log_path, "a") as fr:
                    report = ", ".join(["EPOCH_REPORT", exp_id, "%.4f" % val_loss])
                    print(report)
                    fr.write(report + "\n")

            # 全エポックでもっともval_lossが低くなったモデルのfスコアを計算してモデルを保存
            npz_path = os.path.join(self.trainer.npz_dir, "%s.npz" % exp_id)
            serializers.save_npz(npz_path, min_val_model)

            if type(val_dataset) == list:
                f1_scores = [self.trainer.test(val_dataset) for val_dataset in val_datasets]
                ave_score = sum(f1_scores) / len(f1_scores)
            else:
                ave_score = self.trainer.test(val_dataset)

            with open(self.trainer.log_path, "a") as fr:
                report = ", ".join(["VALIDATION_REPORT", exp_id, "%.4f" % min_val_loss, "%.4f" % ave_score])
                print(report)
                fr.write(report + "\n")
