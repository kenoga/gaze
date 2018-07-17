# -*- coding: utf-8 -*-

import os, sys
import random
import glob
import json
import copy
from collections import defaultdict

import numpy as np
from PIL import Image, ImageOps

class ImagePath():
    def __init__(self, path):
       # format: 
        self.img_name = os.path.basename(path)
        split = self.img_name.split('.')[0].split('_')
        self.path = path
        self.pid = int(split[0])
        self.glasses = bool(split[1])
        self.place = split[2]
        self.target = int(split[3])
        self.locked = None
        self.mirror = False
        self.for_test = False


def balance_data(imgs, nonlocked_rate):
    assert type(imgs) == list
    assert len(imgs) > 0
    assert nonlocked_rate >= 1
    
    result = []
    pid2imgs = {}
    for img in imgs:
        if img.pid not in pid2imgs:
            pid2imgs[img.pid] = {True: [], False: []}
        if img.for_test:
            result.append(img)
        else:
            pid2imgs[img.pid][img.locked].append(img)
    for pid, imgs in pid2imgs.items():
        locked_imgs = pid2imgs[pid][True]
        nonlocked_imgs = pid2imgs[pid][False]
        if len(nonlocked_imgs) > (len(locked_imgs) * nonlocked_rate):
            nonlocked_imgs = random.sample(nonlocked_imgs, len(locked_imgs) * nonlocked_rate)
        
        result.extend(locked_imgs)
        result.extend(nonlocked_imgs)
    return result


def bulk_data(imgs):
    bulked = []
    for img in imgs:
        bulked.append(img)
        if not img.for_test:
            mirrored = copy.deepcopy(img)
            mirrored.mirror = True
            bulked.append(mirrored)
    return bulked

def load_imgpaths(dataset_dir, test_ids, locked_targets, ignored_targets, places, \
                  train_ids=None, \
                  annotation_dict=None,
                  nonlocked_rate=None,
                  bulking=False):
    assert os.path.exists(dataset_dir)
    assert nonlocked_rate >= 1 or nonlocked_rate is None

    imgpaths = glob.glob(os.path.join(dataset_dir, '*/*.jpg'))
    imgs = [ImagePath(path) for path in imgpaths]
    
    if train_ids:
        imgs = [img for img in imgs if img.pid in train_ids or img.pid in test_ids]

    # delete images annotated as noise data.
    if annotation_dict:
        imgs = [img for img in imgs if img.img_name not in annotation_dict or annotation_dict[img.img_name] not in {'closed-eyes', 'other'}]

    # delete images those targets should be ignored
    imgs = [img for img in imgs if img.target not in ignored_targets]
   
    for img in imgs:
        img.locked = True if img.target in locked_targets else False
    
    for img in imgs:
        img.for_test = True if img.pid in test_ids else False
    
    if bulking:
        imgs = bulk_data(imgs)
    
    if nonlocked_rate:
        imgs = balance_data(imgs, nonlocked_rate)

    train = []
    test = []
    
    for img in imgs:
        if img.for_test:
            test.append(img)
        else:
            train.append(img)

    return train, test


class DatasetLoader():
    def __init__(self, dataset_dir, test_ids, batch_size, block_size, img_size, \
                 locked_targets=[0,1,2,31,32, 40, 50], \
                 ignored_targets={}, \
                 places={"A","B","C","D"}, \
                 train_ids=None, \
                 annotation_path=None, \
                 bulking=False,
                 nonlocked_rate=1):

        annotation_dict = None
        if annotation_path:
            with open(annotation_path, 'r') as fr:
                annotation_dict = json.load(fr)
        self.train_paths, self.test_paths = load_imgpaths(dataset_dir, \
                                                          test_ids, \
                                                          locked_targets, \
                                                          ignored_targets, \
                                                          places, train_ids=train_ids, \
                                                          annotation_dict=annotation_dict,
                                                          nonlocked_rate=nonlocked_rate,
                                                          bulking=bulking)
        self.train_path_pool = None
        self.test_path_pool = None
        
        print("Train Locked Size: %d" % len([0 for path in self.train_paths if path.locked == True]))
        print("Train Nonlocked Size: %d" % len([0 for path in self.train_paths if path.locked == False]))
        print("Test Locked Size: %d" % len([0 for path in self.test_paths if path.locked == True]))
        print("Test Nonlocked Size: %d" % len([0 for path in self.test_paths if path.locked == False]))
        self.batch_size = batch_size
        self.block_size = block_size
        self.img_size = img_size
        #  batch_queue 
        self.train_block = []
        self.test_block = []
        self.bulking = bulking
    
    def init(self, ran=True):
        if ran:
            self.train_path_pool = random.sample(self.train_paths, len(self.train_paths))
            self.test_path_pool = random.sample(self.test_paths, len(self.test_paths))
    
    def load_batch(self, paths, return_paths=False):
        xs = []
        ts = []
        error = []
        
        for path in paths:
            try:
#                 img = Image.open(path).convert('RGB') ## Gray->L, RGB->RGB
                img = Image.open(path.path).convert('L') ## Gray->L, RGB->RGB
                img = ImageOps.equalize(img)
            except:
#                 print("Can't load %s." % path)
                error.append(path)
                continue
            img = img.resize((self.img_size[0], self.img_size[1]))
            
            if path.mirror:
                img = ImageOps.mirror(img)
            
            x = np.array(img, dtype=np.float32)
            x = x / 255.0 ## Normalize [0, 255] -> [0, 1]
            x = x.reshape(1, self.img_size[0], self.img_size[1]) ## Reshape image to input shape of CNN
            
            t = np.array(int(path.locked), dtype=np.int32)
            xs.append(x)
            ts.append(t)
        if return_paths:
            return (np.array(xs).astype(np.float32), np.array(ts).astype(np.int32)), paths

        return np.array(xs).astype(np.float32), np.array(ts).astype(np.int32)
    
    
    def load_block(self, path_pool, paths=False):
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
                block.append(self.load_batch(batch_paths, paths))
            else:
                break
        return block
        
        
    def get_batch(self, test=False, paths=False):
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
            block.extend(self.load_block(path_pool, paths))
#             print("Block loading completed...")

        return block.pop()
    
        
# 一度に全ての画像を読み込むのが無理なので1/10ずつ読み込んで返す．
# pathsは全て読み込めるのでそれをランダマイズしてブロックを作りgeneratorで呼び出されるたびに返すようにする
