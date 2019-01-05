# -*- coding: utf-8 -*-
import chainer
import chainer.links as L
import chainer.functions as F

class RNN(chainer.Chain):
    def __init__(self, n_layers=1, n_in=128, n_hidden=16, n_out=2, dropout=0.5):
        super(RNN, self).__init__()
        with self.init_scope():
            self.l1 = L.NStepRNNReLU(n_layers, n_in, n_hidden, dropout)
            self.l2 = L.Linear(n_hidden, n_out, initialW=chainer.initializers.Normal(scale=0.01))
        self.hx = None
        self.dropout_rate = dropout
            
    def __call__(self, xs):
        batch = len(xs)
        hy, hs = self.l1(self.hx, xs)
        self.hx = hy
        
        concat_hs = F.concat(hs, axis=0)
        concat_ys = self.l2(F.dropout(concat_hs, self.dropout_rate))
        ys = F.split_axis(concat_ys, batch, axis=0)
        return ys
    
    def compute_loss(self, xs, ts):
        """
        loss を計算する
        :param xs: list of arrays
        :param ts: list of arrays
        :param normalize: loss 計算時に平均化するかどうか
        :return: loss
        """
        assert isinstance(xs, list)
        assert isinstance(ts, list)

        batch = len(xs)
        ys = self(xs)
        concat_ys = chainer.functions.concat(ys, axis=0)
        concat_ts = chainer.functions.concat(ts, axis=0)
        loss = F.sum(F.softmax_cross_entropy(concat_ys, concat_ts, reduce="no")) / batch
        return loss
    
    def set_state(self, hx):
        self.hx = hx
        
    def reset_state(self):
        self.set_state(None)





        
class GRU(chainer.Chain):
    def __init__(self, n_layers=1, n_in=128, n_hidden=16, n_out=2, dropout=0.5):
        super(GRU, self).__init__()
        with self.init_scope():
            self.l1 = L.NStepGRU(n_layers, n_in, n_hidden, dropout)
            self.l2 = L.Linear(n_hidden, n_out, initialW=chainer.initializers.Normal(scale=0.01))
        self.hx = None
        self.dropout_rate = dropout
            
    def __call__(self, xs):
        batch = len(xs)
        hy, hs = self.l1(self.hx, xs)
        self.hx = hy
        
        concat_hs = F.concat(hs, axis=0)
        concat_ys = self.l2(F.dropout(concat_hs, self.dropout_rate))
        ys = F.split_axis(concat_ys, batch, axis=0)
        return ys

    def compute_loss(self, xs, ts):
        """
        loss を計算する
        :param xs: list of arrays
        :param ts: list of arrays
        :param normalize: loss 計算時に平均化するかどうか
        :return: loss
        """
        assert isinstance(xs, list)
        assert isinstance(ts, list)

        batch = len(xs)
        ys = self(xs)
        concat_ys = chainer.functions.concat(ys, axis=0)
        concat_ts = chainer.functions.concat(ts, axis=0)
        loss = F.sum(F.softmax_cross_entropy(concat_ys, concat_ts, reduce="no")) / batch
        return loss
    
    def set_state(self, hx):
        self.hx = hx
        
    def reset_state(self):
        self.set_state(None)


class LSTM(chainer.Chain):
    def __init__(self, n_layers=1, n_in=128, n_hidden=16, n_out=2, dropout=0.5):
        super(LSTM, self).__init__()
        with self.init_scope():
            self.l1 = L.NStepLSTM(n_layers, n_in, n_hidden, dropout)
            self.l2 = L.Linear(n_hidden, n_out, initialW=chainer.initializers.Normal(scale=0.01))
        self.hx = None
        self.cx = None
        self.dropout_rate = dropout
            
    def __call__(self, xs):
        batch = len(xs)
        hy, cy, hs = self.l1(self.hx, self.cx, xs)
        self.hx = hy
        self.cx = cy
        
        concat_hs = F.concat(hs, axis=0)
        concat_ys = self.l2(F.dropout(concat_hs, self.dropout_rate))
        ys = F.split_axis(concat_ys, batch, axis=0)
        return ys

    def compute_loss(self, xs, ts):
        """
        loss を計算する
        :param xs: list of arrays
        :param ts: list of arrays
        :param normalize: loss 計算時に平均化するかどうか
        :return: loss
        """
        assert isinstance(xs, list)
        assert isinstance(ts, list)

        batch = len(xs)
        ys = self(xs)
        concat_ys = chainer.functions.concat(ys, axis=0)
        concat_ts = chainer.functions.concat(ts, axis=0)
        loss = F.sum(F.softmax_cross_entropy(concat_ys, concat_ts, reduce="no")) / batch
        return loss
    
    
    def set_state(self, hx, cx):
        self.hx = hx
        self.cx = cx
        
    def reset_state(self):
        self.set_state(None, None)
        
 

class AttentionLSTM(chainer.Chain):
    def __init__(self, n_layers=1, n_in=128, n_hidden=16, n_out=2, window_size=8, dropout=0.5):
        super(AttentionLSTM, self).__init__()
        with self.init_scope():
            self.l1 = L.NStepLSTM(n_layers, n_in, n_hidden, dropout)
            self.l2 = L.Linear(n_hidden, n_out, initialW=chainer.initializers.Normal(scale=0.01))
            self.attention = L.Linear(n_hidden, window_size, initialW=chainer.initializers.Normal(scale=0.01))
        self.hx = None
        self.cx = None
        self.n_hidden = n_hidden
        self.dropout_rate = dropout
        self.window_size = window_size
            
    def __call__(self, xs):
        batch = len(xs)
        hy, cy, hs = self.l1(self.hx, self.cx, xs)
        self.hx = hy
        self.cx = cy
        concat_hs = F.concat(hs, axis=0)
        # 各windowの最後のhsからattetionを計算する
        last_hs = concat_hs[self.window_size-1::self.window_size]
        attention_weight = F.softmax(self.attention(last_hs))
        
        #　reshapeで縦にしてから要素積用にコピー
        concat_attentioned = None
        for i in range(batch):
            attention_weight_for_prod = F.reshape(attention_weight[i], (self.window_size,1))
            attention_weight_for_prod = F.broadcast_to(attention_weight_for_prod, (self.window_size, self.n_hidden))
            attentioned = F.sum(hs[i] * attention_weight_for_prod, axis=0)
            
            if concat_attentioned is None:
                concat_attentioned = F.vstack((F.reshape(attentioned, (1, self.n_hidden))))
            else:
                concat_attentioned = F.vstack((concat_attentioned, F.reshape(attentioned, (1, self.n_hidden))))
            
        concat_ys = self.l2(F.dropout(concat_attentioned, self.dropout_rate))
        ys = F.split_axis(concat_ys, batch, axis=0)
        return ys
    
    def get_attention_weight(self, xs):
        batch = len(xs)
        hy, cy, hs = self.l1(self.hx, self.cx, xs)
        self.hx = hy
        self.cx = cy
        concat_hs = F.concat(hs, axis=0)
        # 各windowの最後のhsからattetionを計算する
        last_hs = concat_hs[self.window_size-1::self.window_size]
        attention_weight = F.softmax(self.attention(last_hs))
        return attention_weight
        

    def compute_loss(self, xs, ts):
        """
        loss を計算する
        :param xs: list of arrays
        :param ts: list of arrays
        :param normalize: loss 計算時に平均化するかどうか
        :return: loss
        """
        assert isinstance(xs, list)
        assert isinstance(ts, list)

        batch = len(xs)
        ys = self(xs)
        concat_ys = chainer.functions.concat(ys, axis=0)
        concat_ts = chainer.functions.concat(ts, axis=0)
                                              
        loss = F.sum(F.softmax_cross_entropy(concat_ys, concat_ts, reduce="no")) / batch
        return loss
    
    def set_state(self, hx, cx):
        self.hx = hx
        self.cx = cx
        
    def reset_state(self):
        self.set_state(None, None)

