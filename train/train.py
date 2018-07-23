# -*- coding: utf-8 -*-

import time
import random
import os, sys
import json
import chainer
import pickle

from model.cnn import CNN, CNNWithFCFeature

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer import cuda

from sklearn.metrics import precision_recall_fscore_support


def train(model, dataloader, model_path, learn_rate=0.01, epoch=20, gpu=1, use_fc_feature=False):
    print(' '.join(['-' * 25, 'training', '-' * 25]))
    print('# gpu: {}'.format(gpu))
    print('# epoch: {}'.format(epoch))
    print('# learnrate: {}'.format(learn_rate))
    
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
            x, t_batch = batches
            
            if type(model) == CNNWithFCFeature:
                x_batch, f_batch = x
                x_batch = cuda.to_gpu(x_batch)
                f_batch = cuda.to_gpu(f_batch)
                y = model(x_batch, f_batch)
            else:
                x_batch = x
                x_batch = cuda.to_gpu(x_batch)
                y= model(x_batch)

            t_batch = cuda.to_gpu(t_batch)
            t = chainer.Variable(t_batch)
            
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
        test_y_all = np.array([])
        test_t_all = np.array([])
        while True:
            with chainer.using_config('train', False):
                batches = dataloader.get_batch(test=True)
                if batches is None:
                    break
                x, t_batch = batches
                
                if type(model) == CNNWithFCFeature:
                    x_batch, f_batch = x
                    x_batch = cuda.to_gpu(x_batch)
                    f_batch = cuda.to_gpu(f_batch)
                    y = model(x_batch, f_batch)
                else:
                    x_batch = x
                    x_batch = cuda.to_gpu(x_batch)
                    y= model(x_batch)
                
                t_batch = cuda.to_gpu(t_batch)
                t = chainer.Variable(t_batch)

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
    print("test_f1_max: %f" % test_f1_max)
    save_result(model_path, results, model)

    
def save_result(model_path, results, model):    
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

    