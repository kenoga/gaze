 # -*- coding: utf-8 -*-

import os, sys
import random
import glob
import json
import copy
from collections import deque

sys.path.append(os.path.abspath('../../'))
from dataste_utils.Dataset import Dataset

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
        
def check_conf(conf):

    
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
        check_conf_val(conf, 'group_num')
        check_conf_val(conf, 'locked_targets')
        check_conf_val(conf, 'bulking')
        self.bulking = conf['bulking']
        check_conf_val(conf, 'nonlocked_rate')
        self.nonlocked_rate = conf['nonlocked_rate'] 
        
        # データセットの分割数はデータセットの人数以下でなければならない
        assert conf['group_num'] <= len(conf['pids'])
        # データセットの分割数は最低でも3 (train, validation, test)
        assert conf['group_num'] >= 3
        self.grouped_pids = group_list(self.pids, self.group_num)
        
        self.test_index = 0
        paths = paths[::self.skip_num+1]
        
        self.dataset = Dataset(conf['dataset_path'])
        
        # データセットのフィルタリング，必要な情報の読み込みなど
        
        if conf['pids']:
            self.dataset.filter_pid(conf['pids'])
        # if conf['']
        
        if conf['noise_data_path']:
            with open(conf['noise_data_path'], 'r') as fr:
                noise_dict = json.load(fr)
                self.dataset.filter_noise(self.noise_dict)

        if conf['annotation_path']:
            with open(conf['noise_data_path'], 'r') as fr:
                noise_dict = json.load(fr)
                self.dataset.filter_noise(self.noise_dict)
            self.dataset.filter_noise2(self.annotation_dict)
            
        if conf['ignored_targets']:
            self.dataset.filter_target(ignored_targets)
            
        if conf['face_direction_path']:
            face_dir_dict = {}
            dir_path = conf['face_direction_dir']
            json_fnames = [fname for fname in os.listdir(dir_path) if 'json' in fname]
            for json_fname in json_fnames:
                path = os.path.join(dir_path, json_fname)
                with open(path, 'r') as fr:
                    d = json.load(fr)
                    for k, v in d.items():
                        face_dir_dict[k] = v
            self.dataset.load_face_direction_feature(face_dir_dict)
    
    def remains(self):
        return True if self.test_index < self.group_num else False

    def get_test_index(self):
        return self.test_index

    def get_paths(self):
        # split num回まで
        if self.test_index >= self.group_num:
            return None
        
        test_ids = self.grouped_pids[self.test_index]
        val_index = self.test_index + 1 if self.test_index + 1 < self.group_num else 0
        val_ids = self.grouped_pids[val_index]
        
        ipaths = self.dataset.data
       
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

          
