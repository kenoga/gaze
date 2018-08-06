# -*- coding: utf-8 -*-

import os, sys
import random
import glob
import json
import copy
from collections import deque

from ImagePath import ImagePath, FujikawaImagePath

def balance_train_data(imgs, nonlocked_rate):
    assert type(imgs) == list
    assert len(imgs) > 0
    assert nonlocked_rate >= 1
    
    result = []
    pid2imgs = {}
    for img in imgs:
        if img.pid not in pid2imgs:
            pid2imgs[img.pid] = {True: [], False: []}
        if img.type != 'train':
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

def bulk_train_data(imgs):
    bulked = []
    for img in imgs:
        bulked.append(img)
        if img.type == 'train':
            mirrored = copy.deepcopy(img)
            mirrored.mirror = True
            bulked.append(mirrored)
    return bulked


def check_conf_val(config, key):
    assert config is not None 
    try:    
        assert key in config
    except:
        print(key)
        raise AssertionError
    try:
        assert config[key] is not None
    except:
        print(key)
        raise AssertionError 
def group_list(li, group_num):
    # listをgroup_num個に分割する
    # その際groupに含まれる要素の数の差が小さくなるように分割する
    result = []
    
    dq = deque()
    for e in li:
        dq.append(e)
    li = dq
        
    size = len(li)
    while group_num > 0:
        tmp = []
        group_size = int(size / group_num)
        for _ in range(group_size):
            tmp.append(dq.popleft())
        result.append(tmp)
        size -= group_size
        group_num -= 1
    
    return result
        
class DataPathProvider():

    def __init__(self, conf):
        check_conf_val(conf, 'dataset_path')
        
        check_conf_val(conf, 'pids')
        self.pids = conf['pids']
        check_conf_val(conf, 'group_num')
        self.split_num = conf['group_num']
        check_conf_val(conf, 'ignored_targets')
        self.ignored_targets = conf['ignored_targets']
        check_conf_val(conf, 'places')
        self.places = conf['places']
        check_conf_val(conf, 'bulking')
        self.bulking = conf['bulking']
        check_conf_val(conf, 'nonlocked_rate')
        self.nonlocked_rate = conf['nonlocked_rate']
        check_conf_val(conf, 'skip_num')
        self.skip_num = conf['skip_num']
        
        if 'annotation_path' in conf and conf['annotation_path'] is not None:
            with open(annotation_path, 'r') as fr:
                self.annotation_dict = json.load(fr)
        else:
            self.annotation_dict = None
            
        if 'face_dir_dict' in conf and conf['face_dir_dict'] is not None: 
            self.use_face_dir_feature = True
            self.face_dir_dict = face_direction_dict
        else:
            self.use_face_dir_feature = False
        
        # データセットの分割数はデータセットの人数以下でなければならない
        assert conf['group_num'] <= len(conf['pids'])
        # データセットの分割数は最低でも3 (train, validation, test)
        assert conf['group_num'] >= 3
        
        self.grouped_pids = group_list(self.pids, self.split_num)
        
        self.test_index = 0
        paths = sorted(glob.glob(os.path.join(conf['dataset_path'], '*/*.jpg')))
        paths = paths[::self.skip_num+1]
        ipaths = [ImagePath(path) for path in paths]
        ipaths = [ipath for ipath in ipaths if ipath.pid in self.pids]
        
        # delete images annotated as noise data.
        if self.annotation_dict:
            ipaths = [ipath for ipath in ipaths \
                    if ipath.img_name not in annotation_dict \
                    or self.annotation_dict[ipath.img_name] not in {'closed-eyes', 'other'}]

        # delete images those targets should be ignored
        ipaths = [ipath for ipath in ipaths if ipath.target not in self.ignored_targets]

        self.ipaths = ipaths
    def get_paths(self):
        # split num回まで
        if self.test_index >= self.split_num:
            return None
        
        test_ids = self.grouped_pids[self.test_index]
        val_index = self.test_index + 1 if self.test_index < self.split_num else 0
        val_ids = self.grouped_pids[val_index]
        
        ipaths = self.ipaths
       
        for ipath in ipaths:
            ipath.locked = True if ipath.target in self.locked_targets else False
        
        for ipath in ipaths:
            if ipath.pid in test_ids:
                ipath.type = 'test'
            elif ipath.pid in val_ids:
                ipath.type = 'validation'
            else:
                ipath.type = 'train'
        
        if self.bulking:
            ipaths = bulk_train_data(ipaths)
        
        if self.nonlocked_rate:
            ipaths = balance_train_data(ipaths, nonlocked_rate)

        train = []
        validation = []
        test = []
        
        for ipath in ipaths:
            if ipath.type == 'test':
                test.append(ipath)
            elif ipath.type == 'validation':
                validation.append(ipath)
            else:
                train.append(ipath)
        
        self.test_index += 1
        
        # report dataset details
        print(' '.join(['-' * 25, 'dataset', '-' * 25]))
        print("test ids: " + ",".join([str(pid) for pid in test_ids]))
        print("validation ids: " + ",".join([str(pid) for pid in val_ids]))
        print("train ids: " + ",".join([str(pid) for pid in self.pids if pid not in test_ids and pid not in val_ids]))
        print("train locked size: %d" % len([0 for path in train if path.locked == True]))
        print("train nonlocked size: %d" % len([0 for path in train if path.locked == False]))
        print("validation locked size: %d" % len([0 for path in validation if path.locked == True]))
        print("validation nonlocked size: %d" % len([0 for path in validation if path.locked == False]))
        print("test locked size: %d" % len([0 for path in test if path.locked == True]))
        print("test nonlocked size: %d" % len([0 for path in test if path.locked == False]))
        print("all locked sizes: %d" % len([0 for path in ipaths if path.locked == True]))
        print("all unlocked sizes: %d" % len([0 for path in ipaths if path.locked == False]))
        
        
        return train, validation, test
    
    def report(self):
        pass

          
