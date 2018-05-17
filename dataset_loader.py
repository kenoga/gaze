# -*- coding: utf-8 -*-

import os
import random
import glob

from PIL import Image




def paths2dict(paths, locked_targets):
    # 0 or 1
    def path2label(path, locked_targets):
        target = int(path.split(".")[0].split("_")[3])
        return int(target in path)
    
    def path2pid(path):
        return int(path.split(".")[0].split("_")[0])
    
    d = {}
    for path in paths:
        pid = path2pid(path)
        label = path2label(path, locked_targets)
        if pid not in d:
            d[pid] = {}
            d[pid][1] = []
            d[pid][0] = []
        d[pid][label].append(path)
    return d

def balance_locked_and_nonlocked(pid2paths, nonlocked_rate):
    # TODO: randomに多いほうからサンプリングする
    assert type(pid2paths) == dict
    assert nonlocked_rate >= 1
    
    for pid, paths in pid2paths:
        locked_paths = paths[0]
        nonlocked_paths = paths[1]
        paths[1] = ramdom.sample(nonlocked_paths, len(locked_paths) * nonlocked_rate)
        
    return pid2paths

def separate_train_and_test(pid2paths, test_ids):
    train = {}
    test = {}
    for pid, paths in pid2paths:
        if pid in test_ids:
            test[pid] = paths
        else:
            train[pid] = paths
    
    return train, test

def make_tuple_dataset(pid2paths):
    xs = []
    ts = []
    
    for pid, paths in pid2paths:
        for label, paths in paths.items():
            for path in paths:
                img = Image.open(path).convert('RGB') ## Gray->L, RGB->RGB
                img = img.resize((INPUT_WIDTH, INPUT_HEIGHT))

                x = np.array(img, dtype=np.float32)
                ## Normalize [0, 255] -> [0, 1]
                x = x / 255.
                ## Reshape image to input shape of CNN
                x = x.transpose(2, 0, 1)
                #x = x.reshape(3, INPUT_HEIGHT, INPUT_WIDTH)
                
                t = np.array(label, dtype=np.int32)
                
                xs.append(x)
                ts.append(t)
            
    return np.array(xs).astype(np.float32), np.array(ts).astype(np.int32)


def load_dataset(dataset_dir, test_ids, nonlocked_rate=1, locked_targets={0, 1, 2, 31, 32}):
    assert os.exists(dataset_dir)
    assert nonlocked_rate >= 1
    
    imgpaths = glob.glob(dataset_dir + '*/*.jpg')
    assert len(imgpaths) > 1000
    
    locked_vs_nonlocked_rate = 1
    
    # person_id => (locked_paths, nonlocked_paths)のdictを作成
    pid2paths = paths2dict(imgpaths, locked_targets)
    
    # 各人物におけるlocked_pathsとnonlocked_pathsの割合をランダムで調整する
    pid2paths = balance_locked_and_nonlocked(pid2paths, nonlocked_rate)
    
    # 引数で与えられたtest_idsを元にdictをtrainとtestに分割する
    train_pid2paths, test_pid2paths = separate_train_and_test(pid2paths, test_ids)
    
    # 画像を読み込んでndarrayのデータセットを作成する
    x_train, t_train = make_dataset(train_pid2paths)
    x_test, t_test = make_dataset(test_pid2paths)
    
    print('# Train locked images: {}'.format(len(train_locked_paths)))
    print('# Train unlocked images: {}'.format(len(train_unlocked_paths)))
    print('# Test locked images: {}'.format(len(test_locked_paths)))
    print('# Test unlocked images: {}\n'.format(len(test_unlocked_paths)))
    
    # if shuffle: random.shuffle(train)
    return x_train, t_train, x_test, t_test