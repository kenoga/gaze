# -*- coding: utf-8 -*-

import os
import random
import glob

import numpy as np
from PIL import Image, ImageOps


def paths2dict(paths, locked_targets):
    def path2label(path, locked_targets):
        target = int(os.path.basename(path).split(".")[0].split("_")[3])
        if target in locked_targets:
            return 1
        else:
            return 0 
    
    def path2pid(path):
        pid = int(os.path.basename(path).split(".")[0].split("_")[0])
        return pid
    
    d = {}
    for path in paths:
        pid = path2pid(path)
        label = path2label(path, locked_targets)
        if pid not in d:
            d[pid] = {0: [], 1: []}
        d[pid][label].append(path)
    return d

def ignore_targets(paths, targets):
    result = []
    for path in paths:
        target = int(os.path.basename(path).split(".")[0].split("_")[3])
        if target not in targets:
            result.append(path)
    return result


def balance_locked_and_nonlocked(pathslist, nonlocked_rate):
    assert type(pathslist) == list
    assert nonlocked_rate >= 1
    
    result = []
    for paths in pathslist:
        locked_paths = paths[1]
        nonlocked_paths = paths[0]
        if len(nonlocked_paths) > (len(locked_paths) * nonlocked_rate):
            nonlocked_paths = random.sample(nonlocked_paths, len(locked_paths) * nonlocked_rate)
        result.append({1: locked_paths, 0: nonlocked_paths})
    return result


def separate_train_and_test(pid2paths, test_ids, train_ids=None):
    train = []
    test = []
    for pid, allpaths in pid2paths.items():
        if pid in test_ids:
            test.append(allpaths)
        else:
            if train_ids is None or pid in train_ids:
                train.append(allpaths)
                
        
#         for label, paths in allpaths.items():
#             for path in paths:
#                 if pid in test_ids:
#                     test.append((path, label))
#                 else:
#                     train.append((path, label))
    return train, test


def load_imgpaths(dataset_dir, test_ids, nonlocked_rate, locked_targets, ignored_targets, places, train_ids=None):
    assert os.path.exists(dataset_dir)
    assert nonlocked_rate >= 1 or nonlocked_rate is None

    imgpaths = glob.glob(os.path.join(dataset_dir, '*/*.jpg'))
    imgpaths = [path for path in imgpaths if os.path.basename(path).split(".")[0].split("_")[2] in places]
    assert len(imgpaths) > 1
    
    imgpaths = ignore_targets(imgpaths, ignored_targets)
    
    # person_id => {1: locked_paths, 1: nonlocked_paths}のdictを作成
    pid2paths = paths2dict(imgpaths, locked_targets)
    
     # 引数で与えられたtest_idsを元にdictをtrainとtestに分割する
    # それぞれ(locked_paths, nonlocked_paths)のリスト
    trains, tests = separate_train_and_test(pid2paths, test_ids, train_ids=train_ids)

    # 各人物におけるlocked_pathsとnonlocked_pathsの割合をランダムで調整する
    if nonlocked_rate is not None:
        trains = balance_locked_and_nonlocked(trains, nonlocked_rate)
    train = []
    for t in trains:
        # t: {1: locked_paths, 0: nonlocked_paths}
        for label, paths in t.items():
            train.extend([(path, label) for path in paths])
    test = []
    for t in tests:
        # t: {1: locked_paths, 0: nonlocked_paths}
        for label, paths in t.items():
            test.extend([(path, label) for path in paths])

    return train, test




class DatasetLoader():
    def __init__(self, dataset_dir, test_ids, batch_size, block_size, img_size, nonlocked_rate=2, locked_targets={0,1,2,31,32}, ignored_targets={3,4,5,6}, places={"A","B","C","D"}, train_ids=None):
        self.train_paths, self.test_paths = load_imgpaths(dataset_dir, test_ids, nonlocked_rate, locked_targets, ignored_targets, places, train_ids=train_ids,)
        self.train_path_pool = None
        self.test_path_pool = None
#         self.train_path_pool = random.sample(self.train_paths, len(self.train_paths))
#         self.test_path_pool = random.sample(self.test_paths, len(self.test_paths))
        print("Train Locked Size: %d" % len([0 for path in self.train_paths if path[1] == 1]))
        print("Train Nonlocked Size: %d" % len([0 for path in self.train_paths if path[1] == 0]))
        print("Test Locked Size: %d" % len([0 for path in self.test_paths if path[1] == 1]))
        print("Test Nonlocked Size: %d" % len([0 for path in self.test_paths if path[1] == 0]))
        self.batch_size = batch_size
        self.block_size = block_size
        self.img_size = img_size
        #  batch_queue 
        self.train_block = []
        self.test_block = []
    
    def init(self):
        self.train_path_pool = random.sample(self.train_paths, len(self.train_paths))
        self.test_path_pool = random.sample(self.test_paths, len(self.test_paths))
    
    def load_batch(self, paths):
        xs = []
        ts = []
        error = []
        
        for path, label in paths:
            try:
#                 img = Image.open(path).convert('RGB') ## Gray->L, RGB->RGB
                img = Image.open(path).convert('L') ## Gray->L, RGB->RGB
                img = ImageOps.equalize(img)
            except:
#                 print("Can't load %s." % path)
                error.append(path)
                continue
            img = img.resize((self.img_size[0], self.img_size[1]))

            x = np.array(img, dtype=np.float32)
            ## Normalize [0, 255] -> [0, 1]
            x = x / 255.0
            ## Reshape image to input shape of CNN
#             x = x.transpose(2, 0, 1)
            x = x.reshape(1, self.img_size[0], self.img_size[1])
            
            t = np.array(label, dtype=np.int32)
            xs.append(x)
            ts.append(t)
            
        return np.array(xs).astype(np.float32), np.array(ts).astype(np.int32)
    
    
    def load_block(self, path_pool):
        # block_size個のバッチをリストで返す
        assert len(path_pool) > 0
        block = []
        for _ in range(self.block_size):
            batch_paths = []
            for _ in range(self.batch_size):
                if len(path_pool) == 0:
                    break
                batch_paths.append(path_pool.pop())
            if len(batch_paths) > 0:
                block.append(self.load_batch(batch_paths))
            else:
                break
        return block
        
        
    def get_batch(self, test=False):
        if self.train_path_pool is None or self.test_path_pool is None:
            print("Please initialize.")
            return None
        
        if test:
            path_pool = self.test_path_pool
            block = self.test_block
        else:
            path_pool = self.train_path_pool
            block = self.train_block

        # blockが空だったらメモリに読み込む
        if len(block) == 0:
            #  全てのbatchをすでに吐き出していたらNoneを返す
            if len(path_pool) == 0:
                return None
#             print("Block loading now...")
            block.extend(self.load_block(path_pool))
#             print("Block loading completed...")

        return block.pop()
                         
# 一度に全ての画像を読み込むのが無理なので1/10ずつ読み込んで返す．
# pathsは全て読み込めるのでそれをランダマイズしてブロックを作りgeneratorで呼び出されるたびに返すようにする
# batchを返すのもここでやったほうがスッキリかも