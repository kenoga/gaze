# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.path.abspath(".."))
import datetime
import chainer
import cupy as cp
from chainer import optimizers

from training.default_conf import get_default_conf
from network.feedforward import *
from network.rnn import *
from training.trainer.feed_forward_trainer import *
from training.trainer.nstep_rnn_trainer import *
from training.trainer.one_step_rnn_trainer import *
from training.trainer.trainer import CrossValidationTrainerWrapper
from training.data_loader.dataset_loader import CrossValidationDatasetsIterator
from training.data_loader.data_iterator import *

####  Parameter Settings :)
networks = [MultiFrameAttentionOneLayerFeedForwardNeuralNetwork]
batch_sizes = [64]
window_sizes = [2,4,8,16,32]
dataset_path_and_rnn_inputs = [("./dataset/dataset_fc2.pickle", 128, "fc2"), ("./dataset/dataset_fc1.pickle", 256, "fc1")]

####  Loading Default Configuration
conf = get_default_conf()
gpu = conf["gpu"]
train_ids = conf["train_ids"]
test_ids = conf["test_ids"] = [5]
nn_output = conf["nn_output"] = 2
dropout = conf["dropout"] = 0.5
gpu = conf["gpu"] = 0
epoch  = conf["epoch"]
xp = conf["xp"]

#### Output Taget Configuration
output_dir = os.path.join("training", "output", "multi_frame_attention_test_dialog_id_%02d" % conf["test_ids"][0])
npz_dir = os.path.join(output_dir, "npz")
loss_dir = os.path.join(output_dir, "loss")
log_dir = os.path.join(output_dir, "log")
for d in [npz_dir, loss_dir, log_dir]:
    if not os.path.exists(d):
        os.makedirs(d)
log_path = os.path.join(log_dir, "{0:%Y%m%d%H%M%S}.txt".format(datetime.datetime.now()))

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
                dataset_path = dataset_path_and_rnn_input[0]
                input_type = dataset_path_and_rnn_input[2]
                nn_input = dataset_path_and_rnn_input[1]
                
                # network_inputType_batchSize_window_size_trainSize
                config_id =  "%s_%s_%04d_%02d_%02d" % \
                                (network.name, input_type, batch_size, window_size, len(train_ids)-1)
                                 
                network_params = {"n_in": nn_input, "n_out": nn_output, "window_size": window_size}
                
                MultiFrameDataIterator.set_params(batch_size, window_size, xp=cp)
                dataset_iterator = CrossValidationDatasetsIterator(\
                    MultiFrameDataIterator,\
                    dataset_path,\
                    test_ids,\
                    train_ids)
                trainer = FeedForwardTrainer(network, network_params)
                cvtrainer = CrossValidationTrainerWrapper(trainer, dataset_iterator, epoch, log_path, npz_dir, loss_dir, config_id)
                cvtrainer.cross_validate()

