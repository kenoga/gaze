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

import chainer
from chainer.dataset import convert
import chainer.links as L
import chainer.functions as F
from chainer import serializers

from model import CNN


INPUT_WIDTH = 128
INPUT_HEIGHT = 128

## paths to train image directory and test image one
DATASET_PATH = '/home/nogawa/gaze/data/eyecontact_dataset/cg_eyecontact/datasets/entire_face'

## model name for save trained, and model name for  testing
SAVE_MODEL = 'result/model2'

## Train hyper-parameters
CLASS = 2
INPUT_WIDTH = 128
INPUT_HEIGHT = 128
MINIBATCH_SIZE = 8
LEARN_RATE = 0.001
EPOCH = 50
GPU = 1

def load_dataset(dataset_path, shuffle=True):
    filepaths = glob.glob(dataset_path + '/*.jp*g')
    filepaths.sort()
    
    locked_vs_unlocked_rate = 2
    locked_paths = [path for path in filepaths if '_0V_0H' in os.path.basename(path)]
    unlocked_paths = [path for path in filepaths if '_0V_0H' not in os.path.basename(path)]
    
    train_locked_paths = [path for path in locked_paths if int(os.path.basename(path).split('_')[0]) <= 45]
    test_locked_paths = [path for path in locked_paths if int(os.path.basename(path).split('_')[0]) > 45]
    train_unlocked_paths = [path for path in unlocked_paths if int(os.path.basename(path).split('_')[0]) <= 45]
    test_unlocked_paths = [path for path in unlocked_paths if int(os.path.basename(path).split('_')[0]) > 45]
    
    # lockedとunlockedの数を合わせるためにランダムサンプリングする
    train_unlocked_paths = random.sample(train_unlocked_paths, len(train_locked_paths) * 2)
    test_unlocked_paths = random.sample(train_unlocked_paths, len(test_locked_paths) * 2)
    
    def load_image(path):
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
        
        return (x, t)
        
    train = [load_image(path) for path in train_locked_paths + train_unlocked_paths]
    test = [load_image(path) for path in test_locked_paths + test_unlocked_paths]
    
    print('# Train locked images: {}'.format(len(train_locked_paths)))
    print('# Train unlocked images: {}'.format(len(train_unlocked_paths)))
    print('# Test locked images: {}'.format(len(test_locked_paths)))
    print('# Test unlocked images: {}\n'.format(len(test_unlocked_paths)))
    
    if shuffle: random.shuffle(train)
    return train, test
    
    
def main_train(train_model):

    print('\nmodel training start!!\n')
    print('# GPU: {}'.format(GPU))
    print('# Minibatch-size: {}'.format(MINIBATCH_SIZE))
    print('# Epoch: {}'.format(EPOCH))
    print('# Learnrate: {}'.format(LEARN_RATE))

    ## Load train and test images 
    train, test = load_dataset(DATASET_PATH)

    if len(train) < 1 or len(test) < 1:
        raise Exception('train num : {}, test num: {}'.format(len(train), len(test)))

    train_count = len(train)
    test_count = len(test)

    print('# Train images: {}'.format(train_count))
    print('# Test images: {}\n'.format(test_count))

    ## model define
    model = train_model

    ## Set GPU device
    if GPU:
        chainer.cuda.get_device(GPU).use()
        model.to_gpu()

    ## Set Optimizer
    optimizer = chainer.optimizers.MomentumSGD(LEARN_RATE)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    train_iter = chainer.iterators.SerialIterator(train, MINIBATCH_SIZE)
    test_iter = chainer.iterators.SerialIterator(test, MINIBATCH_SIZE, repeat=False, shuffle=False)

    ## Training start!!
    start = time.time()

    print('epoch  train_loss  train_accuracy  test_loss  test_accuracy  Elapsed-Time')

    while train_iter.epoch < EPOCH:

        batch = train_iter.next()
        # Reduce learning rate by 0.5 every 25 epochs.
        #if train_iter.epoch % 25 == 0 and train_iter.is_new_epoch:
        #    optimizer.lr *= 0.5
        #    print('Reducing learning rate to: ', optimizer.lr)

        train_losses = []
        train_accuracies = []

        x_array, t_array = convert.concat_examples(batch, GPU)
        x = chainer.Variable(x_array)
        t = chainer.Variable(t_array)

        y = model(x)
        loss_train = F.softmax_cross_entropy(y, t)
        accuracy_train = F.accuracy(y, t)
        model.cleargrads()
        loss_train.backward()
        optimizer.update()

        train_losses.append(chainer.cuda.to_cpu(loss_train.data))
        accuracy_train.to_cpu()
        train_accuracies.append(accuracy_train.data)

        if train_iter.is_new_epoch:
            #print('epoch: ', train_iter.epoch)
            #print('train mean loss: {:.2f}, accuracy: {:.2f}'.format( sum_loss_train / train_count, sum_accuracy_train / train_count))
            # evaluation

            test_losses = []
            test_accuracies = []

            sum_accuracy_test = 0
            sum_loss_test = 0
            #model.predictor.train = False
            for batch in test_iter:
                x_array, t_array = convert.concat_examples(batch, GPU)
                x = chainer.Variable(x_array)
                t = chainer.Variable(t_array)

                y = model(x)

                loss_test = F.softmax_cross_entropy(y, t)
                accuracy_test = F.accuracy(y, t)

                test_losses.append(chainer.cuda.to_cpu(loss_test.data))
                accuracy_test.to_cpu()
                test_accuracies.append(accuracy_test.data)


            test_iter.reset()
            #model.predictor.train = True

            print('{:>5}  {:^10.4f}  {:^14.4f}  {:^9.4f}  {:^13.4f}  {:^12.2f}'.format(train_iter.epoch, np.mean(train_losses), np.mean(train_accuracies), np.mean(test_losses), np.mean(test_accuracies), time.time()-start))


    print('\ntraining finished!!\n')

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
