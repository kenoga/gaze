
# -*- coding: utf-8 -*-

import os
import cupy as cp
import chainer
import chainer.cuda
from chainer import serializers
import pickle

from training.trainer import TrainerBase
from training.default_conf import get_default_conf
from training.dataset_loader import DatasetLoader
from network.lstm import LSTM, GRU, RNN


def translate_conf(conf):
    if conf["network"] == "gru":
        conf["network"] = GRU
    elif conf["network"] == "lstm":
        conf["network"] = LSTM
    elif conf["network"] == "rnn":
        conf["network"] = RNN
    else:
        raise RuntimeError("invalid conf")
    
    if conf["input_type"] == "fc2":
        conf["rnn_input"] = 128
        conf["dataset_path"] = "./dataset/dataset_fc2.pickle"
    elif conf["input_type"] == "fc3":
        conf["rnn_input"] = 1
        conf["dataset_path"] = "./dataset/dataset_fc3.pickle"
        
    else:
        raise RuntimeError("invalid conf")
    return conf

def confstr2conf(conf_str):
    conf = {}
    conf_str_s = conf_str.split("_")
    conf["network"] = conf_str_s[0]
    conf["input_type"] = conf_str_s[1]
    conf["rnn_hidden"] = int(conf_str_s[2])
    conf["batch_size"] = int(conf_str_s[3])
    conf["window_size"] = int(conf_str_s[4])
    return conf

def test(target_model_path, output_dir):
    # Load conf
    conf = get_default_conf()
    newconf = confstr2conf(os.path.basename(target_model_path).split(".")[0])
    conf.update(newconf)
    conf = translate_conf(conf)

    # Load model
    model = conf["network"](1, conf["rnn_input"], conf["rnn_hidden"], 2, 0.5)
    model.to_gpu()
    chainer.cuda.get_device(0).use()
    serializers.load_npz(target_model_path, model)
    conf["model"] = model
    
    # Load test datasets
    dataset_loader = DatasetLoader(conf["dataset_path"], conf["batch_size"], conf["window_size"], cp)
    test_datasets = dataset_loader.load_by_dialog_id(5)
    
    tester = TrainerBase(conf)
    
    # Test
    for i, test_dataset in enumerate(test_datasets):
        print(test_dataset)
        with chainer.using_config('train', True):
            score, (ts_all, ys_all) = tester.test(test_dataset, True)
        print(score)
        ys_all = chainer.functions.softmax(ys_all)
        with open(os.path.join(output_dir, "%02d_%02d_%s.pickle" % (test_dataset.dialog_id, test_dataset.session_id, test_dataset.seat_id)), "w") as fw:
            pickle.dump((chainer.cuda.to_cpu(ts_all.data), chainer.cuda.to_cpu(ys_all.data)), fw)

# change data size
# if __name__ == "__main__":
#     target_files = ["lstm_fc2_0128_08_32_01_02.npz", "lstm_fc2_0128_08_32_02_02.npz", "lstm_fc2_0128_08_32_03_02.npz"]
#     target_dir = os.path.join(".", "training", "output", "test_dialog_id_05_change_train_data_size", "npz")
#     for target_file in target_files:
#         print(target_file)
#         train_data_size = target_file.split("_")[5]
#         output_dir = os.path.join("evaluation", "test_result", "data_size_%s" % train_data_size)
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         target_path = os.path.join(target_dir, target_file)
#         test(target_path, output_dir)
    
# GRU and RNN
# if __name__ == "__main__":
#     target_files = ["gru_fc2_0064_16_16_04.npz", "rnn_fc2_0064_08_32_04.npz"]
#     target_dir = os.path.join(".", "training", "output", "test_dialog_id_05", "npz")
#     for target_file in target_files:
#         print(target_file)
#         output_dir = os.path.join("evaluation", "test_result", "model_compare_%s" % target_file.split("_")[0])
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         target_path = os.path.join(target_dir, target_file)
#         test(target_path, output_dir)

# fc3
if __name__ == "__main__":
    target_files = ["lstm_fc3_0016_08_16_04.npz"]
    target_dir = os.path.join(".", "training", "output", "test_dialog_id_05", "npz")
    for target_file in target_files:
        print(target_file)
        output_dir = os.path.join("evaluation", "test_result", "fc3")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        target_path = os.path.join(target_dir, target_file)
        test(target_path, output_dir)
    