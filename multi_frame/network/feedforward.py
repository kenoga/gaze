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
#             self.l1 = L.Linear(n_in, n_out)

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
