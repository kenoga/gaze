# -*- coding: utf-8 -*-
import os, sys
import json
import argparse
import pickle
from collections import defaultdict

import chainer

from model.cnn import CNN, CNNWithFCFeature
# from dataset_loader import DatasetLoader
# from dataset_loader import DataPathProvider, BatchProvider
from utils.DataPathProvider import DataPathProvider
from utils.BatchProvider import BatchProvider

import train


from sklearn.metrics import precision_recall_fscore_support

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
        
def main(conf_id):
    conf_def = load_conf('./init.json', 'default')
    conf = load_conf('./init.json', conf_id, conf=conf_def)
    conf['conf_id'] = conf_id
    report_conf(conf)
    
    model = CNN(2)
    
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
    
    # cross validation
    while path_provider.remains():
        if face_dir_dict:
            model = CNNWithFCFeature(2)
        else:
            model = CNN(2)
        
        result_path = os.path.join(conf['result_path'], conf_id + ("_%02d" % path_provider.get_test_index()))
        model_path = os.path.join(conf['model_path'], conf_id + ("_%02d" % path_provider.get_test_index()))
        train_paths, validation_paths, test_paths = path_provider.get_paths()
        
        batch_provider = BatchProvider(
            train_paths,
            validation_paths,
            test_paths,
            conf['batch_size'], 
            conf['block_size'], 
            conf['img_size'], 
            face_dir_dict=face_dir_dict
        )

        train.train(model, batch_provider, result_path, model_path, epoch=conf['epoch'], learn_rate=conf['learn_rate'], gpu=conf['gpu'])
    
    # print('save all result as json file.')
    # with open(model_path + '.json', 'w') as fw:
    #     json.dump(result, fw, indent=2)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_id")
    args = parser.parse_args()
    main(args.config_id)
