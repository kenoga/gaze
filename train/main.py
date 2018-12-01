# -*- coding: utf-8 -*-

import os, sys
import json
import argparse
import pickle
import random
import numpy as np
import cupy as cp
from collections import defaultdict

import chainer

import train
from model.cnn import *
from utils.DataPathProvider import DataPathProviderForCrossValidation
from utils.BatchProvider import BatchProvider
from utils.DataLoader import *


def load_conf(conf_fpath, conf_id, default_conf_id):
    assert os.path.exists(conf_fpath)

    with open(conf_fpath, "r") as fr:
        all_conf = json.load(fr)
        default_conf = all_conf[default_conf_id]
        conf = all_conf[conf_id]

    # defaultを書き換える
    for key, value in conf.items():
        default_conf[key] = value

    default_conf['conf_id'] = conf_id

    return default_conf

def report_conf(conf):
    print(' '.join(['-' * 25, 'conf', '-' * 25]))
    for key, value in sorted(conf.items()):
        print('%s => %s' % (key, value))

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    cp.random.seed(seed)

def cross_validation(conf):
    set_random_seed(conf["random_seed"])

    path_provider = DataPathProviderForCrossValidation(conf)

    GivenDataLoader = globals()[conf["data_loader"]]
    data_loader = GivenDataLoader(conf["img_size"])

    cnn = globals()[conf["model"]]

    for expi in range(conf["exp_num"]):
        fname = "_".join([conf_id, "%02d"%expi])

        path_provider.init()

        # cross validation
        result_all = {}
        while path_provider.remains():
            model = cnn()

            result_path = os.path.join(conf['result_path'], fname, "%02d" % path_provider.get_test_index())
            model_path = os.path.join(conf['model_path'], fname, "%02d" % path_provider.get_test_index())

            train_paths, validation_paths, test_paths = path_provider.get_paths()

            batch_provider = BatchProvider(
                data_loader,
                train_paths,
                validation_paths,
                test_paths,
                conf['batch_size'],
                conf['block_size'],
                conf['img_size']
            )

            result = train.train_and_test(model, batch_provider, result_path, model_path, \
                epoch=conf['epoch'], learn_rate=conf['learn_rate'], gpu=gpu)
            index = path_provider.get_test_index()
            result_all[index] = result

        result_path = os.path.join(conf['result_path'], fname, 'all.json')
        print('save all results as .json --> {}'.format(result_path))
        with open(result_path, 'w') as fw:
            json.dump(result_all, fw, indent=2)

def cross_validation(conf):
    set_random_seed(conf["random_seed"])

    path_provider = DataPathProviderForCrossValidation(conf)

    GivenDataLoader = globals()[conf["data_loader"]]
    data_loader = GivenDataLoader(conf["img_size"])

    cnn = globals()[conf["model"]]

    for expi in range(conf["exp_num"]):
        fname = "_".join([conf_id, "%02d"%expi])

        path_provider.init()

        # cross validation
        result_all = {}
        while path_provider.remains():
            model = cnn()

            result_path = os.path.join(conf['result_path'], fname, "%02d" % path_provider.get_test_index())
            model_path = os.path.join(conf['model_path'], fname, "%02d" % path_provider.get_test_index())

            train_paths, validation_paths, test_paths = path_provider.get_paths()

            batch_provider = BatchProvider(
                data_loader,
                train_paths,
                validation_paths,
                test_paths,
                conf['batch_size'],
                conf['block_size'],
                conf['img_size']
            )

            result = train.train_and_test(model, batch_provider, result_path, model_path, \
                epoch=conf['epoch'], learn_rate=conf['learn_rate'], gpu=gpu)
            index = path_provider.get_test_index()
            result_all[index] = result

        result_path = os.path.join(conf['result_path'], fname, 'all.json')
        print('save all results as .json --> {}'.format(result_path))
        with open(result_path, 'w') as fw:
            json.dump(result_all, fw, indent=2)

def train(conf):
    set_random_seed(conf["random_seed"])

    path_provider = DataPathProviderForCrossValidation(conf)

    GivenDataLoader = globals()[conf["data_loader"]]
    data_loader = GivenDataLoader(conf["img_size"])

    cnn = globals()[conf["model"]]

    for expi in range(conf["exp_num"]):
        fname = "_".join([conf_id, "%02d"%expi])
        model = cnn()

        result_path = os.path.join(conf['result_path'], fname, "%02d" % path_provider.get_test_index())
        model_path = os.path.join(conf['model_path'], fname, "%02d" % path_provider.get_test_index())
        train_paths, validation_paths, test_paths = path_provider.get_paths()

        batch_provider = BatchProvider(
            data_loader,
            train_paths,
            validation_paths,
            test_paths,
            conf['batch_size'],
            conf['block_size'],
            conf['img_size']
        )

        result = train.train_and_test(model, batch_provider, result_path, model_path, \
            epoch=conf['epoch'], learn_rate=conf['learn_rate'], gpu=gpu)

        result_path = os.path.join(conf['result_path'], fname, 'all.json')
        print('save all results as .json --> {}'.format(result_path))
        with open(result_path, 'w') as fw:
            json.dump(result, fw, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_id")
    parser.add_argument("--gpu", default=0)
    args = parser.parse_args()

    conf_id = args.config_id
    conf = load_conf('./init.json', conf_id, 'default')
    conf['gpu'] = args.gpu
    report_conf(conf)
    if conf['type'] == 'cross_validation':
        cross_validation(conf)
    elif conf['type'] == 'train':
        train(conf)
