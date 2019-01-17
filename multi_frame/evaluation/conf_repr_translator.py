

import inspect
from network import rnn
from network import feedforward
from network.rnn import *
from network.feedforward import *

class ConfigRepresentationTranslator(object):
    def __init__(self):
        self.networks = []
        networks.extend(self._get_classes(rnn))
        networks.extend(self._get_classes(feedforward))
        self.repr2network = {}
        for nw in networks:
            self.repr2network[nw.name] = nw

    def _get_classes(self, module):
        return map(lambda x:x[0], inspect.getmembers(module, inspect.isclass))

    def _input_type2input_size(self, input_type):
        if input_type = "fc1":
            return 256
        elif input_type == "fc2":
            return 128
        elif input_type == "fc3":
            return 1

    def _input_type2dataset_path(self, input_type):
        if input_type = "fc1":
            return "./dataset/dataset_fc1.pickle"
        elif input_type == "fc2":
            return "./dataset/dataset_fc2.pickle"
        elif input_type == "fc3":
            return "./dataset/dataset_fc3.pickle"
        else:
            raise RuntimeError("Unknow input_type %s" % str(input_type))

    def repr2conf(self, repr):
        repr = repr.split("_")

        conf = {}
        conf["network"] = self.repr2network[repr[0]]
        conf["input_type"] = repr[1]
        conf["n_in"] = self._input_type2input_size(repr[1])
        conf["dataset_path"] = repr[1]

        if conf["network"] in {OneLayerFeedForwardNeuralNetwork, TwoLayerFeedForwardNeuralNetwork]:
            conf["batch_size"] = int(repr[2])
        elif conf["network"] in {MultiFrameOneLayerFeedForwardNeuralNetwork, MultiFrameTwoLayerFeedForwardNeuralNetwork, MultiFrameAttentionOneLayerFeedForwardNeuralNetwork}:
            conf["batch_size"] = int(repr[2])
            conf["window_size"] = int(repr[3])
        elif conf["network"] in {RNN, GRU, LSTM, AttentionGRU, AttentionLSTM}:
            conf["n_hidden"] = int(repr[2])
            conf["batch_size"] = int(repr[3])
            conf["window_size"] = int(repr[4])
        elif conf["network"] in {OneStepAttentionLSTM}:
            conf["n_hidden"] = int(repr[2])
            conf["split_num"] = int(repr[3])
            conf["window_size"] = int(repr[4])
        else:
            raise RuntimeError("Unknown model. %s" % conf["network"])
        return conf

    def repr2setting(self, repr):
        '''
        network, network_params, iterator, iterator_params, trainer(tester)を返す
        '''
        conf = self.repr2conf(repr)

        network_params = {}
        iterator_params = {}

        network = conf["network"]
        if network in {OneLayerFeedForwardNeuralNetwork, TwoLayerFeedForwardNeuralNetwork]:
            iterator = SingleFrameDataIterator
            iterator_params = {"batch_size": conf["batch_size"]}
            network_params = {"n_in": conf["n_in"]}
            trainer = FeedForwardTrainer

        elif network in {MultiFrameOneLayerFeedForwardNeuralNetwork, MultiFrameTwoLayerFeedForwardNeuralNetwork}:
            iterator = MultiFrameDataIterator
            iterator_params = {"batch_size": conf["batch_size"], "window_size": conf["window_size"]}
            network_params = {"n_in": conf["n_in"] * conf["window_size"]}
            trainer = FeedForwardTrainer

        elif network in {MultiFrameAttentionOneLayerFeedForwardNeuralNetwork}:
            iterator = MultiFrameDataIterator
            iterator_params = {"batch_size": conf["batch_size"], "window_size": conf["window_size"]}
            network_params = {"n_in": conf["n_in"], "window_size": conf["window_size"]}
            trainer = FeedForwardTrainer

        elif network in {RNN, GRU, LSTM, AttentionGRU, AttentionLSTM}:
            iterator = NStepRNNDataIterator
            iterator_params = {"batch_size": conf["batch_size"], "window_size": conf["window_size"]}
            network_params = {"n_in": conf["n_in"], "n_hidden": conf["n_hidden"]}
            trainer = NStepRNNTrainer

        elif network in {OneStepAttentionLSTM}:
            iterator = OneStepRNNDataIterator
            iterator_params = {"batch_size": conf["batch_size"], "window_size": conf["window_size"]}
            network_params = {"n_in": conf["n_in"], "n_hidden": conf["n_hidden"], "window_size": conf["window_size"]}
            trainer = OneStepRNNTrainer

        else:
            raise RuntimeError("Unknown model. %s" % network)
        return network, network_params, iterator, iterator_params, trainer, conf["dataset_path"]
