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


def main_train(model, dataloader, model_path, learn_rate=0.01, epoch=20, gpu=1):
    print('\nmodel training start!!\n')
    print('# gpu: {}'.format(gpu))
    print('# Epoch: {}'.format(epoch))
    print('# Learnrate: {}'.format(learn_rate))
    
    ## Set gpu device
    if gpu is not None:
        cuda.get_device(gpu).use()
        model.to_gpu()

    ## Set Optimizer
    # optimizer = chainer.optimizers.MomentumSGD(learn_rate)
    optimizer = chainer.optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    ## Training start!!
    start = time.time()

    print('epoch  train_loss  train_accuracy  test_loss  test_accuracy  test_precision  test_recall  test_fscore  Elapsed-Time')
    
    results = {}
    results['train'] = {}
    results['train']['loss'] = []
    results['train']['accuracy'] = []
    results['test'] = {}
    results['test']['loss'] = []
    results['test']['accuracy'] = []
    results['test']['precision'] = []
    results['test']['recall'] = []
    results['test']['fscore'] = []
    test_f1_max = 0
    
    for epoch_i in range(epoch):
        # initialize data loader
        dataloader.init()

        # train
        
        train_losses = []
        train_accuracies = []
        train_f1 = []
        
        while True:
            # train
            batches = dataloader.get_batch(test=False)
            if batches is None:
                break
            x, t = batches
            x_batch = cuda.to_gpu(x)
            t_batch = cuda.to_gpu(t)

            x = chainer.Variable(x_batch)
            t = chainer.Variable(t_batch)
            y = model(x)
            
            loss_train = F.softmax_cross_entropy(y, t)
            accuracy_train = F.accuracy(y, t)
            f1_train = F.f1_score(y, t)
            
            model.cleargrads()
            loss_train.backward()
            optimizer.update()

            train_losses.append(cuda.to_cpu(loss_train.data))
            train_accuracies.append(cuda.to_cpu(accuracy_train.data))
            train_f1.append(cuda.to_cpu(f1_train[0].data))
        
        # test
        test_losses = []
        test_accuracies = []
        test_p = []
        test_r = []
        test_f = []
        test_y_all = np.array([])
        test_t_all = np.array([])
        
        sum_accuracy_test = 0
        sum_loss_test = 0
        #model.predictor.train = False
        
        while True:
            with chainer.using_config('train', False):
                batches = dataloader.get_batch(test=True)
                if batches is None:
                    break
                x, t = batches
                x_batch = cuda.to_gpu(x)
                t_batch = cuda.to_gpu(t)
                x = chainer.Variable(x_batch)
                t = chainer.Variable(t_batch)

                y = model(x, train=False)
                loss_test = F.softmax_cross_entropy(y, t)
                accuracy_test = F.accuracy(y, t)
                argmax_y = np.argmax(y.data, axis=1)
                test_y_all = np.hstack((test_y_all, cuda.to_cpu(argmax_y)))
                test_t_all = np.hstack((test_t_all, cuda.to_cpu(t.data)))

                test_losses.append(cuda.to_cpu(loss_test.data))
                test_accuracies.append(cuda.to_cpu(accuracy_test.data))
        precision, recall, fscore, _ = precision_recall_fscore_support(test_t_all, test_y_all, average='binary')
        test_f1_max = max(test_f1_max, fscore)
        print(test_t_all[:20])
        print(test_y_all[:20])
        print('{:>5}  {:^10.4f}  {:^14.4f}  {:^9.4f}  {:^13.4f}  {:^14.4f}  {:^11.4f}  {:^11.4f}  {:^12.2f}'.format( \
                                                                                   epoch_i, \
                                                                                   np.mean(train_losses), \
                                                                                   np.mean(train_accuracies), \
                                                                                   np.mean(test_losses), \
                                                                                   np.mean(test_accuracies), \
                                                                                   np.mean(precision),
                                                                                   np.mean(recall),
                                                                                   np.mean(fscore),
                                                                                   time.time()-start))
        results['train']['loss'].append(float(np.mean(train_losses)))
        results['train']['accuracy'].append(float(np.mean(train_accuracies)))
        results['test']['loss'].append(float(np.mean(test_losses)))
        results['test']['accuracy'].append(float(np.mean(test_accuracies)))
        results['test']['precision'].append(float(np.mean(precision)))
        results['test']['recall'].append(float(np.mean(recall)))
        results['test']['fscore'].append(float(np.mean(fscore)))
    print('\ntraining finished!!\n')


    ## Save the model and the optimizer
    print("test_f1_max: %f" % test_f1_max)
    print('save model start!!\n')
    directory = model_path.split('/')[0]
    if not os.path.exists(directory):
        os.system('mkdir {}'.format(directory))
        print('make outout model directory --> {}'.format(directory))
        
    print('save all results as json file.')
    with open(model_path + '.json', 'w') as fw:
        json.dump(results, fw, indent=2)
    serializers.save_npz(model_path + '.npz', model)
    print('save the model --> {}'.format(model_path + '.npz') )
    serializers.save_npz(model_path + '.state', optimizer)
    print('save the optimizer --> {}'.format(model_path + '.state'))
    pickle.dump(model, open(model_path + '.pkl', 'wb'))
    print('save the model --> {}'.format(model_path + '.pkl'))

    print('\nmodel save finished!!\n')


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

    main_train(model, dataloader, RESULT_FILE, epoch=conf['epoch'], learn_rate=conf['learn_rate'], gpu=conf['gpu'])

    
def main_different_train_size(conf_id):
    conf_def = load_conf('./init.json', 'default')
    conf = load_conf('./init.json', conf_id, conf=conf_def)
    for key, value in conf.items():
        print('%s => %s' % (key, value))
    
    assert os.path.exists(conf['dataset_path'])
    
    TRAINS = [[4],[4,5],[4,5,6],[4,5,6,7], [4,5,6,7,8,9], [4,5,6,7,8,9,10,11], [4,5,6,7,8,9,10,11,12,13], [4,5,6,7,8,9,10,11,12,13,14,15],[4,5,6,7,8,9,10,11,12,13,14,16,17]]
    
    for train_ids in TRAINS:
        print(train_ids)
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
            bulking=conf['bulking'],
            train_ids=train_ids
            )
        model = CNN(2)
        RESULT_FILE = "./results/%s_train%s" % (conf_id, '-'.join([str(train_id) for train_id in train_ids]))
        print(RESULT_FILE)
        main_train(model, dataloader, RESULT_FILE, epoch=conf['epoch'], learn_rate=conf['learn_rate'], gpu=conf['gpu'])
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_id")
    args = parser.parse_args()
#     main(args.config_id)
    main(args.config_id)