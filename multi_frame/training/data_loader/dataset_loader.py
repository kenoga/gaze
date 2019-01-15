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


class CrossValidationDatasetsIterator(object):
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
