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

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def forward(dataloader, model, purpose, optimizer=None):
    assert purpose in {"train", "validation", "test"}
    
    losses = []
    if purpose == "train":
        accuracies = []
    else:
        y_all = np.array([])
        t_all = np.array([])
        paths_all = []
    
    while True:
        # train
        require_paths = False if purpose == "train" else True
        batches = dataloader.get_batch(dtype=purpose, paths=require_paths)
        if batches is None:
            break
        if purpose == "train":
            x, t_batch = batches
        else:
            batches, paths = batches
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
        
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)

        if purpose == "train":
            model.cleargrads()
            loss.backward()
            optimizer.update()
            accuracies.append(cuda.to_cpu(accuracy.data))
        else:
            argmax_y = np.argmax(y.data, axis=1)
            y_all = np.hstack((y_all, cuda.to_cpu(argmax_y)))
            t_all = np.hstack((t_all, cuda.to_cpu(t.data)))
            paths_all.extend(paths)
        losses.append(cuda.to_cpu(loss.data))
    
    loss = np.mean(losses)
    if purpose == "train":
        accuracy = np.mean(accuracies)
        return loss, accuracy
    else:
        accuracy = accuracy_score(t_all, y_all)
        precision, recall, fscore, _ = precision_recall_fscore_support(t_all, y_all, average='binary')
        return (loss, accuracy), (precision, recall, fscore), (t_all.tolist(), y_all.tolist(), paths)


def train_and_test(model, dataloader, result_path, model_path, learn_rate=0.01, epoch=20, gpu=1, use_fc_feature=False):
    print(' '.join(['-' * 25, 'training and validation', '-' * 25]))
    print('# epoch: {}'.format(epoch))
    # print('# learnrate: {}'.format(learn_rate))
    
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
    
    result = {}
    result['train'] = {}
    result['train']['loss'] = []
    result['train']['accuracy'] = []
    result['val'] = {}
    result['val']['loss'] = []
    result['val']['accuracy'] = []
    result['val']['precision'] = []
    result['val']['recall'] = []
    result['val']['fscore'] = []
    result['test'] = {}
    result['test']['imgs'] = []
    result['test']['y'] = []
    result['test']['t'] = []
    result['test']['miss'] = []
    
    best_model = None
    best_score = None
    updated = False
    
    print('epoch  train_loss  train_accuracy  val_loss  val_accuracy  val_precision  val_recall  val_fscore  updated  Elapsed-Time')
    for epoch_i in range(epoch):
        # initialize data loader
        dataloader.init()
        
        with chainer.using_config('train', True):
            train_loss, train_accuracy = forward(dataloader, model, "train", optimizer)
        with chainer.using_config('train', False):
            (val_loss, val_accuracy), (val_precision, val_recall, val_fscore), _ = forward(dataloader, model, "validation")

        if best_score is None or val_fscore > best_score :
            best_score = val_fscore
            best_model = copy.deepcopy(model)
            updated = True
        else:
            updated = False
            
        
        print('{:>5}  {:^10.4f}  {:^14.4f}  {:^8.4f}  {:^12.4f}  {:^13.4f}  {:^10.4f}  {:^10.4f}  {:^7s}  {:^12.2f}' \
              .format( \
                epoch_i, \
                np.mean(train_loss), \
                np.mean(train_accuracy), \
                np.mean(val_loss), \
                np.mean(val_accuracy), \
                np.mean(val_precision),
                np.mean(val_recall),
                np.mean(val_fscore),
                str(updated),
                time.time()-start))
        
        result['train']['loss'].append(float(train_loss))
        result['train']['accuracy'].append(float(train_accuracy))
        result['val']['loss'].append(float(val_loss))
        result['val']['accuracy'].append(float(val_accuracy))
        result['val']['precision'].append(float(val_precision))
        result['val']['recall'].append(float(val_recall))
        result['val']['fscore'].append(float(val_fscore))
    
    print("the best score in validation set: %f" % best_score)
    
    print(' '.join(['-' * 25, 'test', '-' * 25]))
    with chainer.using_config('train', False):
        (test_loss, test_accuracy), (test_precision, test_recall, test_fscore), (t, y, paths) = forward(dataloader, best_model, "test")

    print("loss: %f" % test_loss)
    print("accuracy: %f" % test_accuracy)
    print("precision: %f" % test_precision) 
    print("recall: %f" % test_recall) 
    print("fscore: %f" % test_fscore)
    result['test']['loss'] = float(test_loss)
    result['test']['accuracy'] = float(test_accuracy)
    result['test']['precision'] = float(test_precision)
    result['test']['recall'] = float(test_recall)
    result['test']['fscore'] = float(test_fscore)
    result['test']['t'] = t
    result['test']['y'] = y
    result['test']['paths'] = paths
    
    save_result(result_path, result)
    save_model(model_path, model)
    return result


def save_result(result_path, result):
    print('save the result as .json --> {}'.format(result_path + '.json') )
    dirpath = os.path.dirname(result_path)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    with open(result_path + '.json', 'w') as fw:
        json.dump(result, fw, indent=2)


def save_model(model_path, model):        
    print('save the model as .npz --> {}'.format(model_path + '.npz') )
    dirpath = os.path.dirname(model_path)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    serializers.save_npz(model_path + '.npz', model)
    #     print('save the model as .pkl --> {}'.format(model_path + '.pkl'))
    #     pickle.dump(model, open(model_path + '.pkl', 'wb'))
    #     print('save the optimizer --> {}'.format(model_path + '.state'))
    #     serializers.save_npz(model_path + '.state', optimizer)

    
