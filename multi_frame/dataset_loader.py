# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import pickle
import chainer

sys.path.append("..")
sys.path.append("../..")
import dataset_utils.utils as utils
import dataset_utils.config as config



class DatasetLoader(object):
    def __init__(self, dataset_path, batch_size, window_size, xp=np):
        self.datasets = pickle.load(open(dataset_path))
        self.batch_size = batch_size
        self.window_size = window_size
        self.xp = xp
        
    def load(self, dialog_id, session_id, seat_id):
        assert dialog_id in self.datasets
        assert seat_id in self.datasets[dialog_id]
        xs, ts = self.datasets[dialog_id][seat_id]
        return DataIterator(xs, ts, self.batch_size, self.window_size, self.xp)
    
    def load_by_dialog_id(self, dialog_id):
        assert dialog_id in self.datasets
        data_iters = []
        for session_id in self.datasets[dialog_id]:
            for seat_id in self.datasets[dialog_id][session_id]:
                data_iters.append(self.load(dialog_id, session_id, seat_id))


class DatasetsIteratorForCrossValidation(object):
    # test_dialog_idで指定されたデータはtestデータなので返さない
    # testデータで評価したい場合は上のDatasetLoaderでload_by_dialog_idでデータを読み込む
    def __init__(self, dataset_path, batch_size, window_size, test_dialog_id, xp=np):
        self.datasets = pickle.load(open(dataset_path))
        self.dialog_ids = [did for did in sorted(self.datasets.keys()) if did != test_dialog_id]
        self.batch_size = batch_size
        self.window_size = window_size
        self.test_dialog_id = test_dialog_id
        self.xp = xp

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
                iterator = DataIterator(xs, ts, self.batch_size, self.window_size, self.xp)
                iterator.set_info(dialog_id, session_id, seat_id)
                iterators.append(iterator)
        return iterators
                
        
class DataIterator(object):
    def __init__(self, xs, ts, batch_size, window_size, xp=np):
        self.xp = xp
        self.batch_size = batch_size
        self.window_size = window_size
        self.xs_all, section_size_x = self._separate_by_batch_size(xs, batch_size)
        self.ts_all, section_size_y = self._separate_by_batch_size(ts, batch_size)
        assert len(self.xs_all) == len(self.ts_all) == batch_size
        for i in range(batch_size):
            assert len(self.xs_all[i])== len(self.ts_all[i])
        assert len(set([len(xs) for xs in self.xs_all])) == 1
        assert len(set([len(ts) for ts in self.ts_all])) == 1
        assert section_size_x == section_size_y
        self.section_size = section_size_x
        
    def __iter__(self):
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

    
