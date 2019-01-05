# -*- coding: utf-8 -*-

import os
import datetime
import cupy as cp
from training.dataset_loader import DataIterator


def get_default_conf():
    '''
    #### Default Config
    conf["rnn_layer"]
    conf["rnn_output"]
    conf["dropout"] 
    conf["gpu"]
    conf["epoch"]
    conf["data_iterator"] 
   
    
    #### Non-Default Config
    conf["network"]
    conf["dataset_path"] 
    conf["rnn_input"] 
    conf["input_type"] 
    conf["batch_size"] 
    conf["window_size"]
    conf["rnn_hidden"]
    conf["result_path"]
    conf["log_path"]
    '''
    conf = {}
    conf["rnn_layer"] = 1
    conf["rnn_output"] = 2
    conf["dropout"] = 0.5
    conf["gpu"] = 0
    conf["epoch"] = 10
    conf["xp"] = cp
    conf["data_iterator"] = DataIterator
    return conf

