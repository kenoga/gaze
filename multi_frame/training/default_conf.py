# -*- coding: utf-8 -*-

import os
import datetime
import cupy as cp


def get_default_conf():
    conf = {}
    conf["train_ids"] = [1,2,3,4]
    conf["test_ids"] = [5]
    conf["nn_output"] = 2
    conf["dropout"] = 0.5
    conf["gpu"] = 0
    conf["epoch"] = 10
    conf["xp"] = cp
    return conf

