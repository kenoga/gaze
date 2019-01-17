# -*- coding: utf-8 -*-
import os, sys
import pickle


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
    
    def load_all_as_list(self, dialog_ids):
        '''
        複数の系列をまとめてloadする
        one_stepに使う
        '''
        xs_list = []
        ts_list = []
        for dialog_id in dialog_ids:
            for session_id in sorted(self.datasets[dialog_id].keys()):
                for seat_id in sorted(self.datasets[dialog_id][session_id].keys()):
                    xs, ts = self.datasets[dialog_id][session_id][seat_id]
                    xs_list.append(xs)
                    ts_list.append(ts)
        iterator = self.iterator(xs_list, ts_list)
        return iterator
    
                
class CrossValidationDatasetsIterator(object):
    # test_dialog_idで指定されたデータはtestデータなので返さない
    # testデータで評価したい場合は上のDatasetLoaderでload_by_dialog_idでデータを読み込む
    def __init__(self, iterator, dataset_path, test_ids, train_ids, load_all_as_list=False):
        self.dataset_loader = DatasetLoader(dataset_path, iterator)
        self.dialog_ids = [did for did in sorted(self.dataset_loader.datasets.keys())\
                            if did not in test_ids and did in train_ids]
        self.test_ids = test_ids
        self.train_ids = train_ids
        self.iterator = iterator
        self.load_all_as_list = load_all_as_list

    def __iter__(self):
        self.val_i = 0
        self.current_val_dialog_id = self.dialog_ids[self.val_i]
        return self

    def next(self):
        if self.val_i >= len(self.dialog_ids):
            raise StopIteration
        train_datasets, val_datasets = [], []

        val_id = self.dialog_ids[self.val_i]
        self.val_i += 1
        self.current_val_dialog_id = val_id
        
        train_ids = [did for did in self.dialog_ids if did != val_id]
        val_ids = [val_id]
        
        if self.load_all_as_list:
            train_datasets = [self.dataset_loader.load_all_as_list(train_ids)]
            val_datasets = [self.dataset_loader.load_all_as_list(val_ids)]
            return train_datasets, val_datasets
        
        for train_id in train_ids:
            train_datasets.extend(self.dataset_loader.load_by_dialog_id(train_id))
        for val_id in val_ids:
            val_datasets.extend(self.dataset_loader.load_by_dialog_id(val_id))
        return train_datasets, val_datasets


if __name__ == "__main__":
    dsl  = DatasetLoader("./dataset/dataset_fc2.pickle", 64, 16, xp=cp, iterator=NormalDataIterator)
    iter = dsl.load(5, 1, "A").__iter__()
