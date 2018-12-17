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
    def __init__(self, dataset_path):
        self.datasets = pickle.load(open(dataset_path))
        
    def load(self, dialog_id, session_id, seat_id, batch_size, window_size, xp=np):
        assert dialog_id in self.datasets
        assert seat_id in self.datasets[dialog_id]
        xs, ts = self.datasets[dialog_id][seat_id]
        return DataIterator(xs, ts, batch_size, window_size, xp)


class DatasetsIteratorForCrossValidation(object):
    def __init__(self, dataset_path, batch_size, window_size, xp=np):
        self.datasets = pickle.load(open(dataset_path))
        self.dialog_ids = sorted(self.datasets.keys())
        self.batch_size = batch_size
        self.window_size = window_size
        self.xp = xp

    def __iter__(self):
        self.test_i = 0
        return self

    def next(self):
        if self.test_i >= len(self.dialog_ids):
            raise StopIteration
        trains, vals, tests = [], [], []

        test_ids = [self.dialog_ids[self.test_i]]
        val_i = (self.test_i + 1) % len(self.dialog_ids)
        val_ids = [self.dialog_ids[val_i]]
        train_ids = [dialog_id for dialog_id in self.dialog_ids if dialog_id not in test_ids and dialog_id not in val_ids]
        
        for did in train_ids:
            trains.extend(self._get_data_iterators(did))
        for did in val_ids:
            vals.extend(self._get_data_iterators(did))
        for did in test_ids:
            tests.extend(self._get_data_iterators(did))
        self.test_i += 1
        return trains, vals, tests

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

    
