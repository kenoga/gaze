
# -*- coding: utf-8 -*-

import os
import cupy as cp
import chainer
import chainer.cuda
from chainer import serializers
import chainer.functions as F
import pickle

from training.trainer import TrainerBase
from training.default_conf import get_default_conf
from training.dataset_loader import DatasetLoader, EachDataIterator
from network.lstm import LSTM, GRU, RNN, AttentionLSTM


def translate_conf(conf):
    if conf["network"] == "gru":
        conf["network"] = GRU
    elif conf["network"] == "lstm":
        conf["network"] = LSTM
    elif conf["network"] == "rnn":
        conf["network"] = RNN
    elif conf["network"] == "atlstm":
        conf["network"] = AttentionLSTM
    else:
        raise RuntimeError("invalid conf")
    
    if conf["input_type"] == "fc2":
        conf["rnn_input"] = 128
        conf["dataset_path"] = "./dataset/dataset_fc2.pickle"
    elif conf["input_type"] == "fc3":
        conf["rnn_input"] = 1
        conf["dataset_path"] = "./dataset/dataset_fc3.pickle"
    elif conf["input_type"] == "fc1":
        conf["rnn_input"] = 256
        conf["dataset_path"] = "./dataset/dataset_fc1.pickle"       
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

def calc_attention(model, dataset):
    ats_all = [None for i in range(dataset.batch_size)]
    for batch in dataset:
        xs, ts = batch
        with chainer.using_config('train', False):
            ats = model.get_attention_weight(xs)

        # バッチごとに分割されているデータを元の順番に戻す
        for batch_i in range(len(ats)):
            reshaped_ats = F.reshape(ats[batch_i], (1, ats[batch_i].shape[0]))
            ats_all[batch_i] = reshaped_ats if ats_all[batch_i] is None else F.vstack((ats_all[batch_i], reshaped_ats))
        
    ats_all = F.concat(ats_all, axis=0)
    return ats_all

def save_attention(target_model_path, output_dir):
    # Load conf
    conf = get_default_conf()
    newconf = confstr2conf(os.path.basename(target_model_path).split(".")[0])
    conf.update(newconf)
    conf = translate_conf(conf)

    # Load model
    if conf["network"] == AttentionLSTM:
        model = conf["network"](1, conf["rnn_input"], conf["rnn_hidden"], 2, conf["window_size"], 0.5)
    else:
        model = conf["network"](1, conf["rnn_input"], conf["rnn_hidden"], 2, 0.5)
    model.to_gpu()
    chainer.cuda.get_device(0).use()
    serializers.load_npz(target_model_path, model)
    conf["model"] = model
    
    # Load test datasets
    dataset_loader = DatasetLoader(conf["dataset_path"], conf["batch_size"], conf["window_size"], cp, iterator=EachDataIterator)
    test_datasets = dataset_loader.load_by_dialog_id(5)
    
    tester = TrainerBase(conf)
    
    # Test
    for i, test_dataset in enumerate(test_datasets):
        print(test_dataset)
        with chainer.using_config('train', False):
            ats = calc_attention(model, test_dataset)
        with open(os.path.join(output_dir, "%02d_%02d_%s.pickle" % (test_dataset.dialog_id, test_dataset.session_id, test_dataset.seat_id)), "w") as fw:
            pickle.dump(chainer.cuda.to_cpu(ats.data), fw)


def save_attention_by_trained_model(model_path, output_dir):
    print("test: %s" % model_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_attention(model_path, output_dir)
 

if __name__ == "__main__":
    target_dir = os.path.join(".", "training", "output", "attention_test_dialog_id_05", "npz")
    model_path = os.path.join(target_dir, "atlstm_fc2_0016_32_32_03_02.npz")
    output_dir = os.path.join("evaluation", "test_result", "attention_weights")
    save_attention_by_trained_model(model_path, output_dir)

    