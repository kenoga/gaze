# -*- coding: utf-8 -*-
import time
import random
import os
import json
import chainer
import glob
import pickle

import numpy as np
from PIL import Image
import sklearn.metrics
import chainer
from chainer.dataset import convert
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer.datasets import TupleDataset
from chainer import cuda

from model import CNN


INPUT_WIDTH = 128
INPUT_HEIGHT = 128

## paths to train image directory and test image one
DATASET_PATH = '/home/nogawa/gaze/data/eyecontact_dataset/cg_eyecontact/datasets/entire_face'
# DATASET_PATH = '../data/entire_face'

## model name for save trained, and model name for  testing
SAVE_MODEL = 'result/model2'

## Train hyper-parameters
CLASS = 2
INPUT_WIDTH = 128
INPUT_HEIGHT = 128
BATCH_SIZE = 8
LEARN_RATE = 0.001
EPOCH = 10
GPU = 1
# GPU = -1

def load_dataset(dataset_path, shuffle=True):
    filepaths = glob.glob(dataset_path + '/*.jp*g')
    filepaths.sort()
    
    locked_vs_unlocked_rate = 1
    locked_paths = [path for path in filepaths if '_0V_0H' in os.path.basename(path)]
    unlocked_paths = [path for path in filepaths if '_0V_0H' not in os.path.basename(path)]
    
    train_locked_paths = [path for path in locked_paths if int(os.path.basename(path).split('_')[0]) <= 45]
    test_locked_paths = [path for path in locked_paths if int(os.path.basename(path).split('_')[0]) > 45]
    train_unlocked_paths = [path for path in unlocked_paths if int(os.path.basename(path).split('_')[0]) <= 45]
    test_unlocked_paths = [path for path in unlocked_paths if int(os.path.basename(path).split('_')[0]) > 45]
    
    # lockedとunlockedの数を合わせるためにランダムサンプリングする
    train_unlocked_paths = random.sample(train_unlocked_paths, len(train_locked_paths) * locked_vs_unlocked_rate)
    test_unlocked_paths = random.sample(test_unlocked_paths, len(test_locked_paths) * locked_vs_unlocked_rate)
    def make_tuple_dataset(paths):
        xs = []
        ts = []
        
        for path in paths:
            img = Image.open(path).convert('RGB') ## Gray->L, RGB->RGB
            img = img.resize((INPUT_WIDTH, INPUT_HEIGHT))

            x = np.array(img, dtype=np.float32)
            ## Normalize [0, 255] -> [0, 1]
            x = x / 255.
            ## Reshape image to input shape of CNN
            x = x.transpose(2, 0, 1)
            #x = x.reshape(3, INPUT_HEIGHT, INPUT_WIDTH)

            ## Get label(ground-truth) from file name path 
            if '_0V_0H' in os.path.basename(path):
                label = 0
            else:
                label = 1
            t = np.array(label, dtype=np.int32)
            
            xs.append(x)
            ts.append(t)
        
        return np.array(xs).astype(np.float32), np.array(ts).astype(np.int32)


    x_train, t_train = make_tuple_dataset(train_locked_paths + train_unlocked_paths)
    x_test, t_test = make_tuple_dataset(test_locked_paths + test_unlocked_paths)
    
    print('# Train locked images: {}'.format(len(train_locked_paths)))
    print('# Train unlocked images: {}'.format(len(train_unlocked_paths)))
    print('# Test locked images: {}'.format(len(test_locked_paths)))
    print('# Test unlocked images: {}\n'.format(len(test_unlocked_paths)))
    
    # if shuffle: random.shuffle(train)
    return x_train, t_train, x_test, t_test
    
    
def main_train(model):

    print('\nmodel training start!!\n')
    print('# GPU: {}'.format(GPU))
    print('# Minibatch-size: {}'.format(BATCH_SIZE))
    print('# Epoch: {}'.format(EPOCH))
    print('# Learnrate: {}'.format(LEARN_RATE))

    ## Load train and test images 
    x_train, t_train, x_test, t_test = load_dataset(DATASET_PATH)
    
    assert(len(x_train) == len(t_train))
    assert(len(x_test) == len(t_test))
    train_num = len(t_train)
    test_num = len(t_test)
    
    if train_num < 1 or test_num < 1:
        raise Exception('train num : {}, test num: {}'.format(len(train), len(test)))
    print('# Train images: {}'.format(train_num))
    print('# Test images: {}\n'.format(test_num))
    
    # 変形
    # x_train = x_train.reshape(len(x_train), 3, INPUT_HEIGHT, INPUT_WIDTH)
    # x_test = x_train.reshape(len(x_test), 3, INPUT_HEIGHT, INPUT_WIDTH)

    ## Set GPU device
    if GPU > 0:
        cuda.get_device(GPU).use()
        model.to_gpu()

    ## Set Optimizer
    # optimizer = chainer.optimizers.MomentumSGD(LEARN_RATE)
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
    
    for epoch_i in range(EPOCH):
        # train
        train_losses = []
        train_accuracies = []
        perm = np.random.permutation(train_num)
        for batch_i in range(0, train_num, BATCH_SIZE):
            x_batch = cuda.to_gpu(x_train[perm[batch_i:batch_i+BATCH_SIZE]])
            t_batch = cuda.to_gpu(t_train[perm[batch_i:batch_i+BATCH_SIZE]])

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
        for batch_i in range(0, test_num, BATCH_SIZE):
            x_batch = cuda.to_gpu(x_test[perm[batch_i:batch_i+BATCH_SIZE]])
            t_batch = cuda.to_gpu(t_test[perm[batch_i:batch_i+BATCH_SIZE]])
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
        results['train']['loss'].append(np.mean(train_losses))
        results['train']['accuracy'].append(np.mean(train_accuracies))
        results['test']['loss'].append(np.mean(test_losses))
        results['test']['loss'].append(np.mean(test_accuracies))
    print('\ntraining finished!!\n')
    
    print('save all results as json file.')
    with open('results.json', 'w') as fw:
        json.dump(results, fw)

    ## Save the model and the optimizer
    print('save model start!!\n')
    directory = SAVE_MODEL.split('/')[0]
    if not os.path.exists(directory):
        os.system('mkdir {}'.format(directory))
        print('make outout model directory --> {}'.format(directory))

    serializers.save_npz(SAVE_MODEL + '.npz', model)
    print('save the model --> {}'.format(SAVE_MODEL + '.npz') )
    serializers.save_npz(SAVE_MODEL + '.state', optimizer)
    print('save the optimizer --> {}'.format(SAVE_MODEL + '.state'))
    pickle.dump(model, open(SAVE_MODEL + '.pkl', 'wb'))
    print('save the model --> {}'.format(SAVE_MODEL + '.pkl'))

    print('\nmodel save finished!!\n')
    
model = CNN(2)
main_train(model)
