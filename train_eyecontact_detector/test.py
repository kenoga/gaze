# -*- coding: utf-8 -*-
import time
import random
import os, sys
import json
import chainer
import pickle
from collections import defaultdict

from neural_networks.cnn import CNN
from dataset_loader import DatasetLoader

import argparse

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer import cuda

from sklearn.metrics import precision_recall_fscore_support


def main_test(model, dataloader, model_path, gpu=0):
    serializers.load_npz("%s.npz" % model_path, model)
    model.to_gpu()
    dataloader.init()
    
    test_losses = []
    test_accuracies = []
    test_y_all = np.array([])
    test_t_all = np.array([])
    
    result = {}
    result['paths'] = []
    result['y'] = []
    result['t'] = []
    result['miss'] = []
    
    start = time.time()
    while True:
        batch_opt = dataloader.get_batch(test=True, paths=True)
        if batch_opt is None:
            break
        batch, paths = batch_opt
        x, t = batch
        x_batch = cuda.to_gpu(x)
        t_batch = cuda.to_gpu(t)

        x = chainer.Variable(x_batch)
        t = chainer.Variable(t_batch)
        y = model(x)
        
        loss_test = F.softmax_cross_entropy(y, t)
        accuracy_test = F.accuracy(y, t)
        argmax_y = np.argmax(y.data, axis=1)
        test_y_all = np.hstack((test_y_all, cuda.to_cpu(argmax_y)))
        test_t_all = np.hstack((test_t_all, cuda.to_cpu(t.data)))
        
        result['paths'].extend([path.img_name for path in paths])
        y = cuda.to_cpu(argmax_y).tolist()
        result['y'].extend(y)
        t = cuda.to_cpu(t.data).tolist()
        result['t'].extend(t)
        miss = [int(y[i]==t[i]) for i in range(len(y))]
        result['miss'].extend(miss)

        test_losses.append(cuda.to_cpu(loss_test.data))
        test_accuracies.append(cuda.to_cpu(accuracy_test.data))
    precision, recall, fscore, _ = precision_recall_fscore_support(test_t_all, test_y_all, average='binary')
    
    result['precision'] = precision
    result['recall'] = recall
    result['fscore'] = fscore
    
    print('{:^14.4f}  {:^9.4f}  {:^13.4f}  {:^14.4f}  {:^11.4f}'.format( \
                                                                                   np.mean(test_losses), \
                                                                                   np.mean(test_accuracies), \
                                                                                   np.mean(precision),
                                                                                   np.mean(recall),
                                                                                 np.mean(fscore)))
    directory = model_path.split('/')[0]
    if not os.path.exists(directory):
        os.system('mkdir {}'.format(directory))
        print('make outout model directory --> {}'.format(directory))
        
    print('save all results as json file.')
    with open(model_path + '_result' '.json', 'w') as fw:
        json.dump(result, fw)
        
        
def load_conf(conf_fpath, conf_id, conf=None):
    assert os.path.exists(conf_fpath)
    
    if conf is None:
        conf = defaultdict(lambda: None)
        
    with open(conf_fpath, 'r') as fr:
        conf_raw = json.load(fr)[conf_id]
    
    for key in conf_raw.keys():
        conf[key] = conf_raw[key]
    
    return conf
    
    
    
def main(conf_id):
    conf_def = load_conf('./init.json', 'default')
    conf = load_conf('./init.json', conf_id, conf=conf_def)
    for key, value in conf.items():
        print('%s => %s' % (key, value))
    
    assert os.path.exists(conf['dataset_path'])
    RESULT_FILE = "./results/%s" % conf_id
    print(RESULT_FILE)
    
    dataloader = DatasetLoader(
        conf['dataset_path'], 
        conf['test_ids'], 
        conf['batch_size'], 
        conf['block_size'], 
        conf['img_size'], 
        places=conf['places'],
        nonlocked_rate=conf['nonlocked_rate'],
        ignored_targets=conf['ignored_targets'],
        annotation_path=conf['annotation_path'],
        bulking=conf['bulking']
        )
    
    model = CNN(2)

    main_test(model, dataloader, RESULT_FILE, gpu=conf['gpu'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_id")
    args = parser.parse_args()
    main(args.config_id)
