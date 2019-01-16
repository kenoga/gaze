# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.path.abspath(".."))
import datetime
import chainer
import cupy as cp
from chainer import optimizers
from network.lstm import *
from training.trainer.one_step_rnn_trainer import OneStepRNNTrainer
from training.trainer.trainer import CrossValidationTrainerWrapper
from training.default_conf import get_default_conf
from training.data_loader.dataset_loader import CrossValidationDatasetsIterator
from training.data_loader.data_iterator import *


####  Parameter Settings :)
networks = [OneStepAttentionLSTM]
split_nums = [16]
window_sizes = [4,8,16,32]
rnn_hiddens = [16, 32, 64, 128]
# dataset_path_and_rnn_inputs = [("./dataset/dataset_fc2.pickle", 128, "fc2"), ("./dataset/dataset_fc1.pickle", 256, "fc1")]
dataset_path_and_rnn_inputs = [("./dataset/dataset_fc2.pickle", 128, "fc2")]

####  Configuration
conf = get_default_conf()
conf["epoch"] = 10
conf["gpu"] = 0
conf["test_ids"] = [5]
conf["train_ids"] = [1,2,3,4]
output_dir = os.path.join("training", "output", "atlstm1step_test_dialog_id_%02d" % conf["test_ids"][0])
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
params = [networks, window_sizes, split_nums, rnn_hiddens, dataset_path_and_rnn_inputs]
exp_size = reduce(lambda x, y: x * y, [len(param) for param in params])
exp_i = 0
for network in networks:
    for rnn_hidden in rnn_hiddens:
        for window_size in window_sizes:
            for split_num in split_nums:
                for dataset_path_and_rnn_input in dataset_path_and_rnn_inputs:
                    exp_i += 1
                    print("exp_i: %d / %d" % (exp_i, exp_size))
                    conf["network"] = network
                    conf["dataset_path"] = dataset_path_and_rnn_input[0]
                    conf["rnn_input"] = dataset_path_and_rnn_input[1]
                    conf["input_type"] = dataset_path_and_rnn_input[2]
                    conf["rnn_hidden"] = rnn_hidden
                    conf["split_num"] = split_num
                    conf["window_size"] = window_size

                    # Setting of Data Iterator
                    OneStepDataIterator.set_params(conf["split_num"], xp=cp)
                    dataset_iterator = CrossValidationDatasetsIterator(\
                        OneStepDataIterator,\
                        conf["dataset_path"],\
                        test_ids=conf["test_ids"],\
                        train_ids=conf["train_ids"],\
                        load_all_as_list=True)

                    # Setting of Trainer
                    trainer = OneStepRNNTrainer(conf)
                    cvtrainer = CrossValidationTrainerWrapper(trainer, dataset_iterator)
                    cvtrainer.cross_validate()
