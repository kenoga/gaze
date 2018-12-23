# -*- coding: utf-8 -*-

import os
import cupy as cp
import chainer
import chainer.cuda
from chainer import serializers
import pickle

from dataset_loader import DatasetLoader
from network.lstm import LSTM, GRU
from best_model_selector import select_best_model_conf
import train_multi_frame_gaze_detector

def translate_conf(conf):
    if conf["network"] == "gru":
        conf["network"] = GRU
    elif conf["network"] == "lstm":
        conf["network"] = LSTM
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


def test(test_dialog_id=5):
    # Select best model
    log_dir = "./training/output/test_dialog_id_05/log"
    log_files = ["train_log_20181219202934.txt", "train_log_20181219205625.txt"]
    
    conf_str = select_best_model_conf(log_dir, log_files)
    conf = confstr2conf(conf_str)
    conf = translate_conf(conf)
        
    model = conf["network"](1, conf["rnn_input"], conf["rnn_hidden"], 2, 0.5)
    model.to_gpu()
    chainer.cuda.get_device(0).use()
    val_id = 4
    
    # Load model
    result_dir = "./result"
    trained_model_path = os.path.join(result_dir, "%s_%02d.npz" % (conf_str, val_id))
    print("load model: %s" % trained_model_path)
    serializers.load_npz(trained_model_path, model)
    
    # Load test datasets
    dataset_loader = DatasetLoader(conf["dataset_path"], conf["batch_size"], conf["window_size"], cp)
    test_datasets = dataset_loader.load_by_dialog_id(5)
    
    # Test
    for i, test_dataset in enumerate(test_datasets):
        print(test_dataset)
        with chainer.using_config('train', True):
            score, (ts_all, ys_all) = train_multi_frame_gaze_detector.test(model, test_dataset, True)
        print(score)
        ys_all = chainer.functions.softmax(ys_all)
        with open("./prediction/%s.pickle" % str(test_dataset), "w") as fw:
            pickle.dump((chainer.cuda.to_cpu(ts_all.data), chainer.cuda.to_cpu(ys_all.data)), fw)
        
    
    
if __name__ == "__main__":
    test(5)