# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import cupy as cp
import pickle
import random
import chainer

class DataIteratorBase(object):
    def set_info(self, dialog_id, session_id, seat_id):
        self.dialog_id = dialog_id
        self.session_id = session_id
        self.seat_id = seat_id

    def __str__(self):
        return "%02d_%02d_%s" % (self.dialog_id, self.session_id, self.seat_id)


class SingleFrameDataIterator(DataIteratorBase):
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
        t_batch = chainer.Variable(self.xp.asarray(self.ts[start: end], dtype=self.xp.int32))
        self.i += 1
        return x_batch, t_batch


class MultiFrameDataIterator(DataIteratorBase):
    # 複数枚の画像を入力とするネットワークを学習する用のデータイテレータ

    batch_size = None
    window_size = None
    xp = None

    @classmethod
    def set_params(cls, batch_size, window_size, xp=cp):
        cls.batch_size = batch_size
        cls.window_size = window_size
        cls.xp = xp

    def __init__(self, xs, ts):
        if self.batch_size == None or self.window_size == None or self.xp == None:
            raise RuntimeError("Essential parameters have not been set.")

        assert len(xs) > 0
        assert len(xs) == len(ts)
        self.xs, self.ts = xs, ts
        # shuffle
#         self.xs, self.ts = self._shuffle(xs, ts)
        self.xdim = len(xs[0])

    def __iter__(self):
        self.window_r = self.window_size - 1
        return self

    def _shuffle(self, xs, ts):
        combined = list(zip(xs, ts))
        random.shuffle(combined)
        xs, ts = zip(*combined)
        return xs, ts

    def next(self):
        '''
        x_batch_list: window_size分のx_batch
        x_batch_list[0]はwindowの一番左をバッチにしたもの
        '''
        x_batch_list = [[] for _ in range(self.window_size)]
        t_batch = []

        if self.window_r >= len(self.xs):
            raise StopIteration

        for window_r_now in range(self.window_r, self.window_r + self.batch_size):
            if window_r_now >= len(self.xs):
                break;
            t_batch.append(self.ts[window_r_now])
            window_l = window_r_now - (self.window_size - 1)
            assert window_l >= 0
            for xs_index, data_index in enumerate(range(window_l, window_r_now + 1)):
                x_batch_list[xs_index].append(self.xs[data_index])
        self.window_r += self.batch_size
        x_batch_list = [chainer.Variable(self.xp.asarray(x_batch, dtype=self.xp.float32)) for x_batch in x_batch_list]
        t_batch = chainer.Variable(self.xp.asarray(t_batch, dtype=self.xp.int32))
        return x_batch_list, t_batch


# # MultiFrameDataIteratorのテスト
# if __name__ == "__main__":
#     from training.data_loader.dataset_loader import DatasetLoader
#     dataset_path = "./dataset/dataset_fc1.pickle"
#     MultiFrameDataIterator.set_params(64, 4)
#     dataset_loader = DatasetLoader(dataset_path, MultiFrameDataIterator)
#     iterator = dataset_loader.load(1, 1, "A").__iter__()
#     x_batch_list, ts = iterator.next()
#     import pdb; pdb.set_trace()




class NStepRNNDataIterator(DataIteratorBase):
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
        if self.batch_size == None or self.window_size == None or self.xp == None:
            raise RuntimeError("Essential parameters have not been set.")

        self.xs_all, section_size_x = self._separate_by_batch_size(xs, self.batch_size)
        self.ts_all, section_size_y = self._separate_by_batch_size(ts, self.batch_size)
        self.section_size = section_size_x
#         assert len(self.xs_all) == len(self.ts_all) == self.batch_size
#         for i in range(self.batch_size):
#             assert len(self.xs_all[i])== len(self.ts_all[i])
#         assert len(set([len(xs) for xs in self.xs_all])) == 1
#         assert len(set([len(ts) for ts in self.ts_all])) == 1
#         assert section_size_x == section_size_y


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


class NStepRNNEachDataIterator(DataIteratorBase):
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
        if self.batch_size == None or self.window_size == None or self.xp == None:
            raise RuntimeError("Essential parameters have not been set.")
        assert len(xs) > 0
        assert len(ts) > 0
        assert len(xs) == len(ts)
        # assert len(set([len(x) for x in xs])) == 1
        # assert len(set([len(t) for t in ts])) == 1
        self.xdim = len(xs[0])
        self.xs_all, section_size_x = self._separate_by_batch_size(xs, self.batch_size)
        self.ts_all, section_size_y = self._separate_by_batch_size(ts, self.batch_size)
        assert len(self.xs_all) == len(self.ts_all) == self.batch_size
        for i in range(self.batch_size):
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

    
class OneStepRNNDataIterator(object):
    '''
    NStepでない通常のRNNのためのイテレータ
    '''
    split_num = None
    xp = None

    @classmethod
    def set_params(cls, split_num, xp=cp):
        cls.split_num = split_num
        cls.xp = xp

    def __init__(self, xs_list, ts_list):
        if self.split_num == None or self.xp == None:
            raise RuntimeError("Essential parameters have not been set.")
        assert len(xs_list) > 0
        assert len(ts_list) > 0
        assert len(xs_list) == len(ts_list)
        self.xdim = len(xs_list[0][0])
        xs_list, ts_list = self._split_seqs(xs_list, self.split_num), self._split_seqs(ts_list, self.split_num)
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

# OneStepDataIteratorのテスト
if __name__ == "__main__":
    from training.data_loader.dataset_loader import DatasetLoader
    dataset_path = "./dataset/dataset_fc2.pickle"
    OneStepDataIterator.set_params(16)
    dataset_loader = DatasetLoader(dataset_path, OneStepDataIterator)
    iterator = dataset_loader.load_all_as_list([1]).__iter__()
    import pdb; pdb.set_trace()
    