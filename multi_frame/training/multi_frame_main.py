# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.path.abspath(".."))
import datetime
import chainer
import cupy as cp
from chainer import optimizers

from training.default_conf import get_default_conf
from network.feedforward import *
from training.trainer.rnn import *
from training.trainer.feed_forward_trainer import *
from training.trainer.nstep_rnn_trainer import *
from training.trainer.one_step_rnn_trainer import *
from training.trainer.trainer import CrossValidationTrainerWrapper
from training.data_loader.dataset_loader import CrossValidationDatasetsIterator
from training.data_loader.data_iterator import *

####  Parameter Settings :)
networks = [MultiFrameOneLayerFeedForwardNeuralNetwork, MultiFrameOneLayerFeedForwardNeuralNetwork]
batch_sizes = [64]
window_sizes = [2,4,8,16,32]
dataset_path_and_rnn_inputs = [("./dataset/dataset_fc2.pickle", 128, "fc2"), ("./dataset/dataset_fc1.pickle", 256, "fc1")]

####  Configuration
conf = get_default_conf()
conf["epoch"] = 10
conf["gpu"] = 1
conf["test_ids"] = [5]
conf["train_ids"] = [1,2,3,4]
output_dir = os.path.join("training", "output", "multi_frame_test_dialog_id_%02d" % conf["test_ids"][0])
npz_dir = os.path.join(output_dir, "npz")
loss_dir = os.path.join(output_dir, "loss")
log_dir = os.path.join(output_dir, "log")
for d in [npz_dir, loss_dir, log_dir]:
    if not os.path.exists(d):
        os.makedirs(d)
conf["npz_dir"] = npz_dir
conf["loss_dir"] = loss_dir
conf["log_path"] = os.path.join(log_dir, "{0:%Y%m%d%H%M%S}.txt".format(datetime.datetime.now()))
print("conf:")
for key, value in sorted(conf.items()):
    print("%s -> %s" % (key, value))

#### Journey to the Best Hiper Parameter Search :)
params = [networks, window_sizes, batch_sizes, dataset_path_and_rnn_inputs]
exp_size = reduce(lambda x, y: x * y, [len(param) for param in params])
exp_i = 0
for network in networks:
    for window_size in window_sizes:
        for batch_size in batch_sizes:
            for dataset_path_and_rnn_input in dataset_path_and_rnn_inputs:
                exp_i += 1
                print("exp_i: %d / %d" % (exp_i, exp_size))
                conf["network"] = network
                conf["dataset_path"] = dataset_path_and_rnn_input[0]
                conf["nn_input"] = dataset_path_and_rnn_input[1] * window_size
                conf["input_type"] = dataset_path_and_rnn_input[2]
                conf["batch_size"] = batch_size
                conf["window_size"] = window_size

                MultiFrameDataIterator.set_params(conf["batch_size"], conf["window_size"], xp=cp)
                dataset_iterator = CrossValidationDatasetsIterator(\
                    MultiFrameDataIterator,\
                    conf["dataset_path"],\
                    test_ids=conf["test_ids"],\
                    train_ids=conf["train_ids"])
                trainer = FeedForwardTrainer(conf)
                cvtrainer = CrossValidationTrainerWrapper(trainer, dataset_iterator)
                cvtrainer.cross_validate()
