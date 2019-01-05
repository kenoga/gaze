# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.path.abspath(".."))
import datetime
import chainer
import cupy as cp
from chainer import optimizers
from network.lstm import LSTM, GRU, RNN, AttentionLSTM, TestAttentionLSTM
from training.trainer import CrossValidationTrainer
from training.default_conf import get_default_conf
from training.dataset_loader import EachDataIterator, BigDataIterator

####  AttentionLSTM :)
networks = [TestAttentionLSTM]
batch_sizes = [8, 16, 32]
dataset_path_and_rnn_inputs = [("./dataset/dataset_fc2.pickle", 128, "fc2")]
window_sizes = [8,16,32]
rnn_hiddens = [16,32,64,128]

conf = get_default_conf()
print("default conf:")
for key, value in sorted(conf.items()):
    print("%s -> %s" % (key, value))

####  Configuration
conf["gpu"] = 1
conf["test_ids"] = [5]
conf["train_ids"] = [1,2,3,4]
output_dir = os.path.join("training", "output", "attention2_test_dialog_id_%02d" % conf["test_ids"][0])
npz_dir = os.path.join(output_dir, "npz")
loss_dir = os.path.join(output_dir, "loss")
log_dir = os.path.join(output_dir, "log")
for d in [npz_dir, loss_dir, log_dir]:
    if not os.path.exists(d):
        os.makedirs(d)
conf["npz_dir"] = npz_dir
conf["loss_dir"] = loss_dir
conf["log_path"] = os.path.join(log_dir, "{0:%Y%m%d%H%M%S}.txt".format(datetime.datetime.now()))
conf["data_iterator"] = BigDataIterator
        
#### Journey to the Best Hiper Parameter Search :)
params = [networks, batch_sizes, dataset_path_and_rnn_inputs, window_sizes, rnn_hiddens]
exp_size = reduce(lambda x, y: x * y, [len(param) for param in params])
exp_i = 0
for network in networks:
    for dataset_path_and_rnn_input in dataset_path_and_rnn_inputs:
        for batch_size in batch_sizes:
            for window_size in window_sizes:
                for rnn_hidden in rnn_hiddens:
                    exp_i += 1
                    print("exp_i: %d / %d" % (exp_i, exp_size))
                    conf["network"] = network
                    conf["dataset_path"] = dataset_path_and_rnn_input[0]
                    conf["rnn_input"] = dataset_path_and_rnn_input[1]
                    conf["input_type"] = dataset_path_and_rnn_input[2]
                    conf["batch_size"] = batch_size
                    conf["window_size"] = window_size
                    conf["rnn_hidden"] = rnn_hidden
                    train = CrossValidationTrainer(conf)
                    train.cross_validate()
