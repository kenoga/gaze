# -*- coding: utf-8 -*-
import time
import random
import os
import json
import chainer
import pickle

from neuralnetworks import CNN
from dataset_loader import load_dataset

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer import cuda


def main_train(model, dataset, model_path, input_size=128, batch_size=8, learn_rate=0.001, epoch=100, gpu=1):
    assert type(dataset) == tuple
    assert len(dataset) == 4
    
    print('\nmodel training start!!\n')
    print('# gpu: {}'.format(gpu))
    print('# Minibatch-size: {}'.format(batch_size))
    print('# Epoch: {}'.format(epoch))
    print('# Learnrate: {}'.format(learn_rate))

    ## Load train and test images 
    x_train, t_train , x_test, t_test = dataset
    assert(len(x_train) == len(t_train))
    assert(len(x_test) == len(t_test))
    train_num = len(t_train)
    test_num = len(t_test)
    
    assert train_num > 0 and test_num > 0

    print('# Train images: {}'.format(train_num))
    print('# Test images: {}\n'.format(test_num))
    
    ## Set gpu device
    if gpu > 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    ## Set Optimizer
    # optimizer = chainer.optimizers.MomentumSGD(learn_rate)
    optimizer = chainer.optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    ## Training start!!
    start = time.time()

    print('epoch  train_loss  train_accuracy  test_loss  test_accuracy  Elapsed-Time')
    
    results = {}
    results['train'] = {}
    results['train']['loss'] = []
    results['train']['accuracy'] = []
    results['test'] = {}
    results['test']['loss'] = []
    results['test']['accuracy'] = []
    
    for epoch_i in range(epoch):
        # train
        train_losses = []
        train_accuracies = []
        perm = np.random.permutation(train_num)
        for batch_i in range(0, train_num, batch_size):
            x_batch = cuda.to_gpu(x_train[perm[batch_i:batch_i+batch_size]])
            t_batch = cuda.to_gpu(t_train[perm[batch_i:batch_i+batch_size]])

            x = chainer.Variable(x_batch)
            t = chainer.Variable(t_batch)

            y = model(x)
            
            loss_train = F.softmax_cross_entropy(y, t)
            accuracy_train = F.accuracy(y, t)
            model.cleargrads()
            loss_train.backward()
            optimizer.update()

            train_losses.append(cuda.to_cpu(loss_train.data))
            accuracy_train.to_cpu()
            train_accuracies.append(accuracy_train.data)
        
        # test
        test_losses = []
        test_accuracies = []
        sum_accuracy_test = 0
        sum_loss_test = 0
        #model.predictor.train = False
            
        perm = np.random.permutation(test_num)
        for batch_i in range(0, test_num, batch_size):
            x_batch = cuda.to_gpu(x_test[perm[batch_i:batch_i+batch_size]])
            t_batch = cuda.to_gpu(t_test[perm[batch_i:batch_i+batch_size]])
            x = chainer.Variable(x_batch)
            t = chainer.Variable(t_batch)

            y = model(x)
            loss_test = F.softmax_cross_entropy(y, t)
            accuracy_test = F.accuracy(y, t)

            test_losses.append(cuda.to_cpu(loss_test.data))
            accuracy_test.to_cpu()
            test_accuracies.append(accuracy_test.data)
            #model.predictor.train = True

        print('{:>5}  {:^10.4f}  {:^14.4f}  {:^9.4f}  {:^13.4f}  {:^12.2f}'.format(epoch_i, np.mean(train_losses), np.mean(train_accuracies), np.mean(test_losses), np.mean(test_accuracies), time.time()-start))
        results['train']['loss'].append(float(np.mean(train_losses)))
        results['train']['accuracy'].append(float(np.mean(train_accuracies)))
        results['test']['loss'].append(float(np.mean(test_losses)))
        results['test']['accuracy'].append(float(np.mean(test_accuracies)))
    print('\ntraining finished!!\n')
    
    print('save all results as json file.')
    print(results)
    with open('results.json', 'w') as fw:
        json.dump(results, fw, indent=2)

    ## Save the model and the optimizer
    print('save model start!!\n')
    directory = model_path.split('/')[0]
    if not os.path.exists(directory):
        os.system('mkdir {}'.format(directory))
        print('make outout model directory --> {}'.format(directory))

    serializers.save_npz(model_path + '.npz', model)
    print('save the model --> {}'.format(model_path + '.npz') )
    serializers.save_npz(model_path + '.state', optimizer)
    print('save the optimizer --> {}'.format(model_path + '.state'))
    pickle.dump(model, open(model_path + '.pkl', 'wb'))
    print('save the model --> {}'.format(model_path + '.pkl'))

    print('\nmodel save finished!!\n')

if __name__ == "__main__":
    # x_train, t_train, x_test, t_test = load_dataset(DATASET_PATH)
    DATASET_PATH = "./data/kobas-omni-dataset/face_image"
    IMG_SIZE = 128
    dataset = load_dataset(DATASET_PATH, IMG_SIZE)
    model = CNN(2)
    main_train(model, dataset, gpu=0)
