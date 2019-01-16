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
                
    
class OneStepDataIterator(object):
    '''

    '''
    def __init__(self, xs_list, ts_list, split_num, xp=np):
        assert len(xs_list) > 0
        assert len(ts_list) > 0
        assert len(xs_list) == len(ts_list)
        self.xp = xp
        self.xdim = len(xs_list[0][0])
        xs_list, ts_list = self._split_seqs(xs_list, split_num), self._split_seqs(ts_list, split_num)
        self.xs_list = chainer.Variable(self._equalize_seq_len(xs_list).data.astype(self.xp.float32))
        self.ts_list = chainer.Variable(self._equalize_seq_len(ts_list).data.astype(self.xp.int32))
        self.seq_len = self.xs_list.shape[1]
        self.batch_size = self.xs_list.shape[0]
        
    def __iter__(self):
        self.i = 0
        return self
    
    def next(self):
        xs, ts = [], []
        
        if self.i >= self.seq_len:
            raise StopIteration
        
        xs = self.xs_list[:, self.i]
#         ts = chainer.functions.reshape(self.ts_list[:, self.i], (self.batch_size, 1))
        ts = self.ts_list[:, self.i]
        
        self.i += 1
        return xs, ts
        
    def _split_seqs(self, seq_list, split_num):
        split_seq_list = []
        for i in range(len(seq_list)):
            split_seq_list.extend(self._split_seq(seq_list[i], split_num))
        return split_seq_list
        
    def _split_seq(self, seq, size):
        newseq = []
        splitsize = 1.0/size*len(seq)
        for i in range(size):
            newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
        return newseq
     
    def _equalize_seq_len(self, seq_list):
        max_seq_len = max([len(seq) for seq in seq_list])
        nd_seq_list = [self.xp.asarray(seq) for seq in seq_list]
        eq_seqs = chainer.functions.pad_sequence(nd_seq_list)
        return eq_seqs
    
    def __str__(self):
        return "%02d_%02d_%s" % (self.dialog_id, self.session_id, self.seat_id)
    

# class DatasetLoader(object):
#     def __init__(self, dataset_path, batch_size, window_size, xp=np, iterator=DataIterator):
#         self.datasets = pickle.load(open(dataset_path))
#         self.batch_size = batch_size
#         self.window_size = window_size
#         self.xp = xp
#         self.iterator = iterator
        
#     def load(self, dialog_id, session_id, seat_id):
#         assert dialog_id in self.datasets
#         assert session_id in self.datasets[dialog_id]
#         assert seat_id in self.datasets[dialog_id][session_id]
#         xs, ts = self.datasets[dialog_id][session_id][seat_id]
#         iterator = self.iterator(xs, ts, self.batch_size, self.window_size, self.xp)
#         iterator.set_info(dialog_id, session_id, seat_id)
#         return iterator
    
#     def load_by_dialog_id(self, dialog_id):
#         assert dialog_id in self.datasets
#         data_iters = []
#         for session_id in self.datasets[dialog_id]:
#             for seat_id in self.datasets[dialog_id][session_id]:
#                 data_iters.append(self.load(dialog_id, session_id, seat_id))
#         return data_iters


# class OneStepDatasetsIteratorForCrossValidation(object):
#     # test_dialog_idで指定されたデータはtestデータなので返さない
#     # testデータで評価したい場合は上のDatasetLoaderでload_by_dialog_idでデータを読み込む
#     def __init__(self, dataset_path, split_num, test_ids, train_ids, xp=np, iterator=OneStepDataIterator):
#         self.datasets = pickle.load(open(dataset_path))
#         self.dialog_ids = [did for did in sorted(self.datasets.keys()) if did not in test_ids and did in train_ids] 
#         self.split_num = split_num
#         self.test_ids = test_ids
#         self.train_ids = train_ids
#         self.xp = xp
#         self.iterator = iterator

#     def __iter__(self):
#         self.val_i = 0
#         self.current_val_dialog_id = self.dialog_ids[self.val_i]
#         return self

#     def next(self):
#         if self.val_i >= len(self.dialog_ids):
#             raise StopIteration
#         train_datasets, val_datasets = [], []

#         val_id = self.dialog_ids[self.val_i]
#         self.current_val_dialog_id = val_id
        
#         train_iterator = self._get_data_iterator(self.train_ids)
#         val_iterator = self._get_data_iterator([val_id])
#         self.val_i += 1
#         return train_iterator, val_iterator

#     def _get_data_iterator(self, dialog_ids):
#         xs_list = []
#         ts_list = []
#         for dialog_id in dialog_ids:
#             for session_id in sorted(self.datasets[dialog_id].keys()):
#                 for seat_id in sorted(self.datasets[dialog_id][session_id].keys()):
#                     xs, ts = self.datasets[dialog_id][session_id][seat_id]
#                     xs_list.append(xs)
#                     ts_list.append(ts)
#         iterator = self.iterator(xs_list, ts_list, self.split_num, self.xp)
#         return iterator
    
  
if __name__ == "__main__":
    from network.lstm import OneStepAttentionLSTM
    dsl  = OneStepDatasetsIteratorForCrossValidation("./dataset/dataset_fc2.pickle", 16, [1,2,3,4], [5], xp=cp, iterator=OneStepDataIterator).__iter__()
    train_iterator, val_iterator = dsl.next()
    iter = train_iterator.__iter__()
    xs, ts = iter.next()
    model = OneStepAttentionLSTM().to_gpu()
    model(xs)
    import pdb; pdb.set_trace()
