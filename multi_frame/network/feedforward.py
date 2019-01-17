# -*- coding: utf-8 -*-
import chainer
import chainer.links as L
import chainer.functions as F

class OneLayerFeedForwardNeuralNetwork(chainer.Chain):
    name = "ff1"

    def __init__(self, n_in=128, n_out=2):
        super(OneLayerFeedForwardNeuralNetwork, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_in, n_out, initialW=chainer.initializers.Normal(scale=0.01))

    def __call__(self, xs):
        return self.l1(xs)

    def compute_loss(self, xs, ts):
        ys = self(xs)
        loss = F.softmax_cross_entropy(ys, ts)
        return loss


class TwoLayerFeedForwardNeuralNetwork(chainer.Chain):
    name = "ff2"

    def __init__(self, n_in=256, n_out=2):
        n_hidden = 128 # 固定
        super(TwoLayerFeedForwardNeuralNetwork, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_in, n_hidden, initialW=chainer.initializers.Normal(scale=0.01))
            self.l2 = L.Linear(n_hidden, n_out, initialW=chainer.initializers.Normal(scale=0.01))

    def __call__(self, xs):
        return self.l2(F.relu(self.l1(xs)))

    def compute_loss(self, xs, ts):
        ys = self(xs)
        loss = F.softmax_cross_entropy(ys, ts)
        return loss

class MultiFrameOneLayerFeedForwardNeuralNetwork(chainer.Chain):
    name = "multiff1"

    def __init__(self, n_in=256, n_out=2):
        super(MultiFrameOneLayerFeedForwardNeuralNetwork, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_in, n_out, initialW=chainer.initializers.Normal(scale=0.01))

    def __call__(self, xs_list):
        xs = F.concat(xs_list)
        return self.l1(xs)

    def compute_loss(self, xs, ts):
        ys = self(xs)
        loss = F.softmax_cross_entropy(ys, ts)
        return loss

class MultiFrameTwoLayerFeedForwardNeuralNetwork(chainer.Chain):
    name = "multiff2"

    def __init__(self, n_in=256, n_out=2):
        n_hidden = 128 # 固定
        super(MultiFrameTwoLayerFeedForwardNeuralNetwork, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_in, n_hidden, initialW=chainer.initializers.Normal(scale=0.01))
            self.l2 = L.Linear(n_hidden, n_out, initialW=chainer.initializers.Normal(scale=0.01))

    def __call__(self, xs_list):
        xs = F.concat(xs_list)
        return self.l2(F.relu(self.l1(xs)))

    def compute_loss(self, xs, ts):
        ys = self(xs)
        loss = F.softmax_cross_entropy(ys, ts)
        return loss

class MultiFrameAttentionOneLayerFeedForwardNeuralNetwork(chainer.Chain):
    name = "atmultiff1"

    def __init__(self, n_in=256, n_out=2, window_size=8):
        super(MultiFrameAttentionOneLayerFeedForwardNeuralNetwork, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_in, n_out, initialW=chainer.initializers.Normal(scale=0.01))
            self.attention = L.Linear(n_in, window_size, initialW=chainer.initializers.Normal(scale=0.01))
        self.n_in = n_in
        self.window_size = window_size

    def __call__(self, xs_list):
        last = xs_list[-1]
        batch = xs_list[0].shape[0]

        # 注意重みを計算 (batch_size, window_size)
        attention_weight = F.softmax(self.attention(last))

        # 注意重みと中間層の履歴で積和をとる
        attentioned = None
        for window_i in range(self.window_size):
            # attention_weight[:, window_i]で列をとる (window_i番目の履歴に対する重みを取得)
            # 要素積を計算するためにreshapeしてからbroadcast_toで敷き詰める
            aw_reshaped = F.reshape(attention_weight[:,window_i], (batch,1))
            aw = F.broadcast_to(aw_reshaped, (batch, self.n_in))
            if attentioned is None:
                attentioned = xs_list[window_i] * aw
            else:
                attentioned += xs_list[window_i] * aw
        return self.l1(attentioned)

    def get_attention_weight(self, xs_list):
        last = xs_list[-1]

        # 注意重みを計算 (batch_size, window_size)
        attention_weight = F.softmax(self.attention(last))

        return attention_weight


    def compute_loss(self, xs, ts):
        ys = self(xs)
        loss = F.softmax_cross_entropy(ys, ts)
        return loss

# class MultiFrameAttentionTwoLayerFeedForwardNeuralNetwork(chainer.Chain):
#     name = "atmultiff2"

#     def __init__(self, n_in=256, n_out=2):
#         n_hidden = 128 # 固定
#         super(MultiFrameTwoLayerFeedForwardNeuralNetwork, self).__init__()
#         with self.init_scope():
#             self.l1 = L.Linear(n_in, n_hidden, initialW=chainer.initializers.Normal(scale=0.01))
#             self.l2 = L.Linear(n_hidden, n_out, initialW=chainer.initializers.Normal(scale=0.01))

#     def __call__(self, xs_list):
#         xs = F.concat(xs_list)
#         return self.l2(F.relu(self.l1(xs)))

#     def compute_loss(self, xs, ts):
#         ys = self(xs)
#         loss = F.softmax_cross_entropy(ys, ts)
#         return loss
