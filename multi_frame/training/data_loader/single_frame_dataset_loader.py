# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import cupy as cp
import pickle
import random
import chainer


class SingleFrameDataIterator(object):
    # 一枚の画像を入力とするネットワークを学習する用のデータイテレータ

    batch_size = None
    xp = None

    @classmethod
    def set_params(cls, batch_size, xp=cp):
        cls.batch_size = batch_size
        cls.xp = xp

    def __init__(self, xs, ts):
        if self.batch_size == None or self.xp == None:
            raise RuntimeError("Essential parameters have not been set.")

        assert len(xs) > 0
        assert len(xs) == len(ts)
        self.xs, self.ts = xs, ts
        # shuffle
#         self.xs, self.ts = self._shuffle(xs, ts)
        self.xdim = len(xs[0])

    def __iter__(self):
        self.i = 0
        return self

    def _shuffle(self, xs, ts):
        combined = list(zip(xs, ts))
        random.shuffle(combined)
        xs, ts = zip(*combined)
        return xs, ts

    def next(self):
        if self.batch_size * self.i >= len(self.xs):
            raise StopIteration
        start = self.batch_size * self.i
        end = self.batch_size * (self.i + 1)
        x_batch = chainer.Variable(self.xp.asarray(self.xs[start: end], dtype=self.xp.float32))
#         t_batch = chainer.Variable(self.xp.asarray(self.ts[start: end], dtype=self.xp.int32))
        t_batch = chainer.Variable(self.xp.asarray(self.ts[start: end], dtype=self.xp.int32))
        self.i += 1
        return x_batch, t_batch

    def set_info(self, dialog_id, session_id, seat_id):
        self.dialog_id = dialog_id
        self.session_id = session_id
        self.seat_id = seat_id

    def __str__(self):
        return "%02d_%02d_%s" % (self.dialog_id, self.session_id, self.seat_id)


class SingleFrameCrossValidationDatasetsIterator(object):
    # test_dialog_idで指定されたデータはtestデータなので返さない
    # testデータで評価したい場合は上のDatasetLoaderでload_by_dialog_idでデータを読み込む
    def __init__(self, iterator, dataset_path, test_ids, train_ids):
        self.datasets = pickle.load(open(dataset_path))
        self.dialog_ids = [did for did in sorted(self.datasets.keys()) if did not in test_ids and did in train_ids]
        self.test_ids = test_ids
        self.train_ids = train_ids
        self.iterator = iterator

    def __iter__(self):
        self.val_i = 0
        self.current_val_dialog_id = self.dialog_ids[self.val_i]
        return self

    def next(self):
        if self.val_i >= len(self.dialog_ids):
            raise StopIteration
        train_datasets, val_datasets = [], []

        val_id = self.dialog_ids[self.val_i]
        self.current_val_dialog_id = val_id

        train_iterator = self._get_data_iterator(self.train_ids)
        val_iterator = self._get_data_iterator([val_id])
        self.val_i += 1
        return train_iterator, val_iterator

    def _get_data_iterator(self, dialog_ids):
        xs = []
        ts = []
        for dialog_id in dialog_ids:
            for session_id in sorted(self.datasets[dialog_id].keys()):
                for seat_id in sorted(self.datasets[dialog_id][session_id].keys()):
                    new_xs, new_ts = self.datasets[dialog_id][session_id][seat_id]
                    xs.extend(new_xs)
                    ts.extend(new_ts)
        iterator = self.iterator(xs, ts)
        return iterator


# if __name__ == "__main__":
#     from network.feedforward import OneLayerFeedForwardNeuralNetwork
#     dsl  = SingleFrameDatasetsIteratorForCrossValidation("./dataset/dataset_fc2.pickle", 64, [1,2,3,4], [5], xp=cp, iterator=SingleFrameDataIterator).__iter__()
#     train_iterator, val_iterator = dsl.next()
#     iter = train_iterator.__iter__()
#     xs, ts = iter.next()
#     model = OneLayerFeedForwardNeuralNetwork(128, 2)
#     model.to_gpu()
#     import pdb; pdb.set_trace()
