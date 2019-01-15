# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import cupy as cp
import pickle
import chainer

sys.path.append("..")
sys.path.append("../..")
import dataset_utils.utils as utils
import dataset_utils.config as config


class NStepDataIterator(object):
    '''
    NStep用のイテレータ
    初期化前に必要なパラメータを設定する
    '''
    batch_size = None
    window_size = None
    xp = None

    @classmethod
    def set_params(cls, batch_size, window_size, xp=cp):
        cls.batch_size = batch_size
        cls.window_size = window_size
        cls.xp = xp

    def __init__(self, xs, ts):
        if cls.batch_size == None or cls.window_sizse == None or cls.xp == None:
            raise RuntimeError("Essential parameters have not been set.")
        self.batch_size = batch_size
        self.window_size = window_size
        self.xp = xp

        self.xs_all, section_size_x = self._separate_by_batch_size(xs, self.batch_size)
        self.ts_all, section_size_y = self._separate_by_batch_size(ts, self.batch_size)
        assert len(self.xs_all) == len(self.ts_all) == batch_size
        for i in range(batch_size):
            assert len(self.xs_all[i])== len(self.ts_all[i])
        assert len(set([len(xs) for xs in self.xs_all])) == 1
        assert len(set([len(ts) for ts in self.ts_all])) == 1
        assert section_size_x == section_size_y
        self.section_size = section_size_x


    def __iter__(self):
        if self.xs_all == None or self.ts_all == None:
            raise RuntimeError("Dataset hasn't been set.")
        self.window_i = 0
        return self

    def next(self):
        xs = []
        ts = []
        i = self.window_size * self.window_i
        if i >= self.section_size:
            raise StopIteration

        for batch_i in range(self.batch_size):
            x = self.xs_all[batch_i][i:i+self.window_size]
            t = self.ts_all[batch_i][i:i+self.window_size]
            xs.append(chainer.Variable(self.xp.asarray(x, dtype=self.xp.float32)))
            ts.append(chainer.Variable(self.xp.asarray(t, dtype=self.xp.int32)))
        self.window_i += 1
        return xs, ts

    def _separate_by_batch_size(self, li, batch_size):
        if len(li) % batch_size == 0:
            virtual_size = len(li)
        else:
            virtual_size = len(li) + batch_size - (len(li) % batch_size)
        section_size = virtual_size / batch_size

        li_s = []
        for i in range(batch_size):
            if i == batch_size - 1:
                li_s.append(li[-section_size:])
            else:
                li_s.append(li[i*section_size:i*section_size+section_size])
        return li_s, section_size

    def set_info(self, dialog_id, session_id, seat_id):
        self.dialog_id = dialog_id
        self.session_id = session_id
        self.seat_id = seat_id

    def __str__(self):
        return "%02d_%02d_%s" % (self.dialog_id, self.session_id, self.seat_id)


class NStepEachDataIterator(object):
    '''
   　windowをひとつずつずらしながらwindow_size分のデータを返していくiterator
    xsがx(1), x(2), x(3), window_sizeが3のときは
    0, 0, x(1) -> 0, x(1), x(2) -> x(1), x(2), x(3)
    のように返す
    '''
    batch_size = None
    window_size = None
    xp = None

    @classmethod
    def set_params(cls, batch_size, window_size, xp=cp):
        cls.batch_size = batch_size
        cls.window_size = window_size
        cls.xp = xp

    def __init__(self, xs, ts):
        if batch_size == None or window_size == None or xp == None:
            raise RuntimeError("Essential parameters have not been set.")
        self.batch_size = batch_size
        self.window_size = window_size
        self.xp = xp
        assert len(xs) > 0
        assert len(ts) > 0
        assert len(xs) == len(ts)
        # assert len(set([len(x) for x in xs])) == 1
        # assert len(set([len(t) for t in ts])) == 1
        self.xdim = len(xs[0])
        self.xs_all, section_size_x = self._separate_by_batch_size(xs, batch_size)
        self.ts_all, section_size_y = self._separate_by_batch_size(ts, batch_size)
        assert len(self.xs_all) == len(self.ts_all) == batch_size
        for i in range(batch_size):
            assert len(self.xs_all[i])== len(self.ts_all[i])
#         assert len(set([len(xs) for xs in self.xs_all])) == 1
#         assert len(set([len(ts) for ts in self.ts_all])) == 1
        assert section_size_x == section_size_y
        self.section_size = section_size_x


    def __iter__(self):
        if self.xs_all == None or self.ts_all == None:
            raise RuntimeError("Dataset hasn't been set.")
        # windowの一番右(最後)のindex
        self.window_r = 0
        return self

    def next(self):
        xs, ts = [], []

        if self.window_r >= self.section_size:
            raise StopIteration

        for batch_i in range(self.batch_size):
            x, t = [], []
            # windowの一番左から右へ
            for i in range(self.window_r - self.window_size + 1, self.window_r + 1):
                if i >= 0:
                    x.append(self.xs_all[batch_i][i])
#                     t.append(self.ts_all[batch_i][i])
                else:
                    x.append(self.xp.zeros(self.xdim, dtype=self.xp.float32))
#                     t.append(0)
            t.append(self.ts_all[batch_i][self.window_r])
            xs.append(chainer.Variable(self.xp.asarray(x, dtype=self.xp.float32)))
            ts.append(chainer.Variable(self.xp.asarray(t, dtype=self.xp.int32)))
        self.window_r += 1
        return xs, ts

    def _separate_by_batch_size(self, li, batch_size):
        if len(li) % batch_size == 0:
            virtual_size = len(li)
        else:
            virtual_size = len(li) + batch_size - (len(li) % batch_size)
        section_size = virtual_size / batch_size

        li_s = []
        for i in range(batch_size):
            if i == batch_size - 1:
                li_s.append(li[-section_size:])
            else:
                li_s.append(li[i*section_size:i*section_size+section_size])
        return li_s, section_size

    def set_info(self, dialog_id, session_id, seat_id):
        self.dialog_id = dialog_id
        self.session_id = session_id
        self.seat_id = seat_id

    def __str__(self):
        return "%02d_%02d_%s" % (self.dialog_id, self.session_id, self.seat_id)


class DatasetLoader(object):
    def __init__(self, dataset_path, iterator):
        self.datasets = pickle.load(open(dataset_path))
        self.iterator = iterator

    def load(self, dialog_id, session_id, seat_id):
        assert dialog_id in self.datasets
        assert session_id in self.datasets[dialog_id]
        assert seat_id in self.datasets[dialog_id][session_id]
        xs, ts = self.datasets[dialog_id][session_id][seat_id]
        iterator = self.iterator(xs, ts)
        iterator.set_info(dialog_id, session_id, seat_id)
        return iterator

    def load_by_dialog_id(self, dialog_id):
        assert dialog_id in self.datasets
        data_iters = []
        for session_id in self.datasets[dialog_id]:
            for seat_id in self.datasets[dialog_id][session_id]:
                data_iters.append(self.load(dialog_id, session_id, seat_id))
        return data_iters


class NStepCrossValidationDatasetIterator(object):
    # test_dialog_idで指定されたデータはtestデータなので返さない
    # testデータで評価したい場合は上のDatasetLoaderでload_by_dialog_idでデータを読み込む
    def __init__(self, iterator, dataset_path, iterator, test_ids, train_ids):
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

        for did in self.dialog_ids:
            if did == val_id:
                val_datasets.extend(self._get_data_iterators(did))
            else:
                train_datasets.extend(self._get_data_iterators(did))
        self.val_i += 1
        return train_datasets, val_datasets

    def _get_data_iterators(self, dialog_id):
        assert dialog_id in self.datasets
        iterators = []
        for session_id in sorted(self.datasets[dialog_id].keys()):
            for seat_id in sorted(self.datasets[dialog_id][session_id].keys()):
                xs, ts = self.datasets[dialog_id][session_id][seat_id]
                iterator = self.iterator(xs, ts)
                iterator.set_info(dialog_id, session_id, seat_id)
                iterators.append(iterator)
        return iterators




if __name__ == "__main__":
    dsl  = DatasetLoader("./dataset/dataset_fc2.pickle", 64, 16, xp=cp, iterator=NormalDataIterator)
    iter = dsl.load(5, 1, "A").__iter__()
