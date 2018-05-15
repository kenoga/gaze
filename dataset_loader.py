# -*- coding: utf-8 -*-

import os
import glob
from PIL import Image

def separate_by_tag(paths):
    locked = []
    nonlocked = []
    for path in paths:
        fname = os.path.basename(path)
        place = fname.split("_")[3]
        if int(place) in {0, 1, 2, 31, 32}:
            locked.append(path)
        else:
            nonlocked.append(path)
    return locked, nonlocked

def balance_size(a, b, random=False):
    # TODO: randomに多いほうからサンプリングする
    assert type(a) == type(b)
    assert type(a) == list
    
    if len(a) == len(b):
        return a, b
    elif len(a) > len(b):
        return a[:len(b)], b
    else:
        return a, b[:len(b)]

def separate_train_and_test(paths, test_rate=0.8):
    train = [path for path in paths[:int(len(locked)*(1-test_rate))]]
    test = [path for path in paths[int(len(locked)*(test_rate)):]]
    return train, test

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
        
def load_dataset(dataset_dir, shuffle=True, test_rate=0.1):
    assert test_rate < 1
    assert os.exists(dataset_dir)
    
    filepaths = glob.glob(dataset_dir + '*/*.jpg')
    assert len(filepaths) > 1000
    
    locked_vs_unlocked_rate = 1
    locked, nonlocked = separate_by_tag(filepath)
    locked, nonlocked = balance_size(locked, nonlocked, random=False)
    assert len(locked) == len(nonlocked)
    
    train_locked, test_locked = separate_train_and_test()
    train_nonlocked, test_nonlocked = separate_train_and_test()
    assert len(train_locked+test_locked) == locked
    assert len(train_nonlocked+test_nonlocked) = nonlocked

    x_train, t_train = make_tuple_dataset(train_locked + train_unlocked)
    x_test, t_test = make_tuple_dataset(test_locked + test_unlocked)
    
    print('# Train locked images: {}'.format(len(train_locked_paths)))
    print('# Train unlocked images: {}'.format(len(train_unlocked_paths)))
    print('# Test locked images: {}'.format(len(test_locked_paths)))
    print('# Test unlocked images: {}\n'.format(len(test_unlocked_paths)))
    
    # if shuffle: random.shuffle(train)
    return x_train, t_train, x_test, t_test