# -*- coding: utf-8 -*-

import os, sys
import json
import argparse
import pickle
from collections import defaultdict

import chainer

import train
from model.cnn import *
from utils.DataPathProvider import DataPathProvider
from utils.BatchProvider import BatchProvider
from utils.DataLoader import *

def load_conf(conf_fpath, conf_id, conf=None):
    assert os.path.exists(conf_fpath)

    if conf is None:
        conf = defaultdict(lambda: None)

    with open(conf_fpath, 'r') as fr:
        conf_raw = json.load(fr)[conf_id]

    for key in conf_raw.keys():
        conf[key] = conf_raw[key]

    return conf

def report_conf(conf):
    print(' '.join(['-' * 25, 'conf', '-' * 25]))
    for key, value in sorted(conf.items()):
        print('%s => %s' % (key, value))

def main(conf_id, gpu=0):
    conf_def = load_conf('./init.json', 'default')
    conf = load_conf('./init.json', conf_id, conf=conf_def)
    conf['conf_id'] = conf_id
    report_conf(conf)

    face_dir_dict = None
    if conf['face_direction_dir']:
        face_dir_dict = {}
        dir_path = conf['face_direction_dir']
        json_fnames = [fname for fname in os.listdir(dir_path) if 'json' in fname]
        for json_fname in json_fnames:
            path = os.path.join(dir_path, json_fname)
            with open(path, 'r') as fr:
                d = json.load(fr)
                for k, v in d.items():
                    face_dir_dict[k] = v


    path_provider = DataPathProvider(conf)
    path_provider.report()

    GivenDataLoader = globals()[conf["data_loader"]]
    data_loader = GivenDataLoader(conf["img_size"])

    cnn = globals()[conf["model"]]

    for expi in range(conf["exp_num"]):
        path_provider.init()

        # cross validation
        result_all = {}
        while path_provider.remains():
            model = cnn()

            result_path = os.path.join(conf['result_path'], conf_id, "%02d" % path_provider.get_test_index())
            model_path = os.path.join(conf['model_path'], conf_id, "%02d" % path_provider.get_test_index())

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

        result_path = os.path.join(conf['result_path'], "_".join([conf_id, "%02d"%expi]), 'all.json')
        print('save all results as .json --> {}'.format(result_path))
        with open(result_path, 'w') as fw:
            json.dump(result_all, fw, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_id")
    parser.add_argument("--gpu", default=0)
    args = parser.parse_args()
    main(args.config_id, args.gpu)
