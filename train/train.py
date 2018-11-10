# -*- coding: utf-8 -*-

import time
import random
import os, sys
import json
import chainer
import pickle
import copy

from model.cnn import *

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer import cuda

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from utils.ExperimentResult import ExperimentResult

def forward(dataloader, model, purpose, optimizer=None):
    assert purpose in {"train", "validation", "test"}

    losses = []
    if purpose == "train":
        accuracies = []
    else:
        y_all = np.array([])
        t_all = np.array([])
        paths_all = []
        prob_all = []

    while True:
        # train
        require_paths = False if purpose == "train" else True
        data = dataloader.get_batch(dtype=purpose, paths=require_paths)
        if data is None:
            break

        if require_paths:
            data, paths = data

        assert len(data) >= 2 # 入力1つ、出力1つは必須なのでチェック
        inputs = list(data[:-1])
        output = data[-1]

        for i in range(len(inputs)):
            inputs[i] = chainer.Variable(cuda.to_gpu(inputs[i]))
        y = model(*inputs)
        t = chainer.Variable(cuda.to_gpu(output))
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
            paths_all.extend([path.name for path in paths])
            prob = F.softmax(y).data.tolist()
            prob_all.extend(prob)
        losses.append(cuda.to_cpu(loss.data))

    loss = np.mean(losses)
    if purpose == "train":
        accuracy = np.mean(accuracies)
        return loss, accuracy
    else:
        accuracy = accuracy_score(t_all, y_all)
        precision, recall, fscore, _ = precision_recall_fscore_support(t_all, y_all, average='binary')
        return (loss, accuracy), (precision, recall, fscore), (t_all.tolist(), y_all.tolist(), paths_all, prob_all)


def train_and_test(model, dataloader, result_path, model_path, learn_rate=0.01, epoch=20, gpu=1, use_fc_feature=False):
    chainer.using_config('cudnn_deterministic', True)

    print("gpu: %d" % gpu)
    print(' '.join(['-' * 25, 'training and validation', '-' * 25]))
    print('# epoch: {}'.format(epoch))
    # print('# learnrate: {}'.format(learn_rate))

    ## Set gpu device
    if gpu is not None:
        cuda.get_device(gpu).use()
        model.to_gpu(gpu)

    ## Set Optimizer
    # optimizer = chainer.optimizers.MomentumSGD(learn_rate)
    optimizer = chainer.optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)
    optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    ## Training start!!
    start = time.time()

    result = ExperimentResult()

    best_model = None
    best_score = None
    updated = False

    result.show_header()
    for epoch_i in range(epoch):
        # initialize data loader
        dataloader.init()1

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

        result.print_tmp_result(epoch_i, train_loss, train_accuracy, val_loss, val_accuracy, val_precision, val_recall, val_fscore, updated, time.time()-start)
        result.add_train_result(train_loss, train_accuracy)
        result.add_validation_result(val_loss, train_accuracy, val_precision, val_recall, val_fscore)


    print("the best score in validation set: %f" % best_score)

    print(' '.join(['-' * 25, 'test', '-' * 25]))
    with chainer.using_config('train', False):
        (test_loss, test_accuracy), (test_precision, test_recall, test_fscore), (t, y, paths, probs) = forward(dataloader, best_model, "test")


    print("loss: %f" % test_loss)
    print("accuracy: %f" % test_accuracy)
    print("precision: %f" % test_precision)
    print("recall: %f" % test_recall)
    print("fscore: %f" % test_fscore)

    result.add_test_result(test_loss, test_accuracy, test_precision, test_recall, test_fscore, t, y, paths, probs)
    save_result(result_path, result)
    save_model(model_path, model)
    return result

def save_result(result_path, result):
    print('save the result as .json --> {}'.format(result_path + '.json') )
    dirpath = os.path.dirname(result_path)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    with open(result_path + '.json', 'w') as fw:
        json.dump(result.content, fw, indent=2)

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
