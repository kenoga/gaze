# -*- coding: utf-8 -*-

import time
import random
import os, sys
import json
import chainer
import pickle
import copy

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

    print('epoch  train_loss  train_accuracy  val_loss  val_accuracy  val_precision  val_recall  val_fscore  Elapsed-Time')
    
    results = {}
    results['train'] = {}
    results['train']['loss'] = []
    results['train']['accuracy'] = []
    results['val'] = {}
    results['val']['loss'] = []
    results['val']['accuracy'] = []
    results['val']['precision'] = []
    results['val']['recall'] = []
    results['val']['fscore'] = []
    results['test'] = {}
    
    best_model = None
    best_score = None
    
    for epoch_i in range(epoch):
        # initialize data loader
        dataloader.init()

        # train
        train_losses = []
        train_accuracies = []
        train_f1 = []
        count = 0
        while True:
            # train
            batches = dataloader.get_batch(dtype='train')
            if batches is None:
                break
            x, t_batch = batches
            
            if type(model) == CNNWithFCFeature:
                x_batch, f_batch = x
                x_batch = cuda.to_gpu(x_batch)
                f_batch = chainer.Variable(cuda.to_gpu(f_batch))
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
        
        # validation
        val_losses = []
        val_accuracies = []
        val_y_all = np.array([])
        val_t_all = np.array([])
        
        while True:
            with chainer.using_config('train', False):
                batches = dataloader.get_batch(dtype='validation')
                if batches is None:
                    break
                x, t_batch = batches
                
                if type(model) == CNNWithFCFeature:
                    x_batch, f_batch = x
                    x_batch = cuda.to_gpu(x_batch)
                    f_batch = chainer.Variable(cuda.to_gpu(f_batch))
                    y = model(x_batch, f_batch)
                else:
                    x_batch = x
                    x_batch = cuda.to_gpu(x_batch)
                    y= model(x_batch)
                
                t_batch = cuda.to_gpu(t_batch)
                t = chainer.Variable(t_batch)

                loss_val = F.softmax_cross_entropy(y, t)
                accuracy_val = F.accuracy(y, t)
                argmax_y = np.argmax(y.data, axis=1)
                val_y_all = np.hstack((val_y_all, cuda.to_cpu(argmax_y)))
                val_t_all = np.hstack((val_t_all, cuda.to_cpu(t.data)))

                val_losses.append(cuda.to_cpu(loss_val.data))
                val_accuracies.append(cuda.to_cpu(accuracy_val.data))
        
        
        precision, recall, fscore, _ = precision_recall_fscore_support(val_t_all, val_y_all, average='binary')
        
        if best_score is None or fscore > best_score :
            best_score = fscore
            best_model = copy.deepcopy(model)
            print("The best model was updated!")

        print(val_t_all[:20])
        print(val_y_all[:20])
        
        print('{:>5}  {:^10.4f}  {:^14.4f}  {:^9.4f}  {:^13.4f}  {:^14.4f}  {:^11.4f}  {:^11.4f}  {:^12.2f}'.format( \
                                                                                   epoch_i, \
                                                                                   np.mean(train_losses), \
                                                                                   np.mean(train_accuracies), \
                                                                                   np.mean(val_losses), \
                                                                                   np.mean(val_accuracies), \
                                                                                   np.mean(precision),
                                                                                   np.mean(recall),
                                                                                   np.mean(fscore),
                                                                                   time.time()-start))
        results['train']['loss'].append(float(np.mean(train_losses)))
        results['train']['accuracy'].append(float(np.mean(train_accuracies)))
        results['val']['loss'].append(float(np.mean(val_losses)))
        results['val']['accuracy'].append(float(np.mean(val_accuracies)))
        results['val']['precision'].append(float(np.mean(precision)))
        results['val']['recall'].append(float(np.mean(recall)))
        results['val']['fscore'].append(float(np.mean(fscore)))
    
    print('\ntraining finished!!\n')
    print("The best score in validation set: %f" % best_score)
    model = best_model
    test_losses = []
    test_accuracies = []
    test_y_all = np.array([])
    test_t_all = np.array([])
    print('test start')
    while True:
        with chainer.using_config('train', False):
            batches = dataloader.get_batch(dtype='test')
            if batches is None:
                break
            x, t_batch = batches

            if type(model) == CNNWithFCFeature:
                x_batch, f_batch = x
                x_batch = cuda.to_gpu(x_batch)
                f_batch = chainer.Variable(cuda.to_gpu(f_batch))
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
    loss = np.mean(test_losses)
    accuracy = np.mean(test_accuracies)
    precision, recall, fscore, _ = precision_recall_fscore_support(test_t_all, test_y_all, average='binary')
    print("loss: %f" % loss)
    print("accuracy: %f" % accuracy)
    print("precision: %f" % precision) 
    print("recall: %f" % recall) 
    print("fscore: %f" % fscore)
    results['test']['loss'] = float(loss)
    results['test']['accuracy'] = float(accuracy)
    results['test']['precision'] = float(precision)
    results['test']['recall'] = float(recall)
    results['test']['fscore'] = float(fscore)
    
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
#     serializers.save_npz(model_path + '.state', optimizer)
#     print('save the optimizer --> {}'.format(model_path + '.state'))
#     pickle.dump(model, open(model_path + '.pkl', 'wb'))
#     print('save the model --> {}'.format(model_path + '.pkl'))
    print('\nmodel save finished!!\n')

    