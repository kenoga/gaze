 # -*- coding: utf-8 -*-

import random
import glob
import json
import copy
from collections import deque
import dataset_utils.Dataset 
import dataset_utils.DataInitiator


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
        self.load_conf_val(conf, 'dataset_path')
        self.load_conf_val(conf, 'group_num')
        self.load_conf_val(conf, 'locked_targets')
        self.load_conf_val(conf, 'bulking')
        self.load_conf_val(conf, 'pids')
        self.load_conf_val(conf, 'nonlocked_rate')
        self.load_conf_val(conf, 'noise_data_path')
        self.load_conf_val(conf, 'annotation_path')
        self.load_conf_val(conf, 'ignored_targets')
        self.load_conf_val(conf, 'face_direction_path')
        self.load_conf_val(conf, 'skip_num')
        self.load_conf_val(conf, 'img_format')
        self.load_conf_val(conf, 'data_initiator')
        self.load_conf_val(conf, 'glasses')


        # データセットの分割数はデータセットの人数以下でなければならない
        assert self.group_num <= len(self.pids)
        # データセットの分割数は最低でも3 (train, validation, test)
        assert self.group_num >= 3
        self.grouped_pids = group_list(self.pids, self.group_num)
        
        self.test_index = 0
        self.dataset = dataset_utils.Dataset.Dataset( \
            self.dataset_path, \
            data_initiator_name=self.data_initiator, \
            img_format=self.img_format)
        # データセットのフィルタリング，必要な情報の読み込みなど
        self.dataset.filter_pid(self.pids)
        self.dataset.set_label(self.locked_targets)
        if self.skip_num:
            self.dataset.skip(self.skip_num)
 
        if self.noise_data_paths:
            for path in self.noise_data_paths:
                with open(path, 'r') as fr:
                    noise_dict = json.load(fr)
                    self.dataset.filter_noise(noise_dict)

        if self.annotation_path:
            with open(self.annotation_path, 'r') as fr:
                self.noise_dict = json.load(fr)
                self.dataset.filter_noise2(self.noise_dict)
            
        if self.ignored_targets:
            self.dataset.filter_target(self.ignored_targets)
        
        if not self.glasses:
            self.dataset.delete_glasses()
            
        if self.face_direction_path:
            face_dir_dict = {}
            dir_path = self.face_direction_dir
            json_fnames = [fname for fname in os.listdir(dir_path) if 'json' in fname]
            for json_fname in json_fnames:
                path = os.path.join(dir_path, json_fname)
                with open(path, 'r') as fr:
                    d = json.load(fr)
                    for k, v in d.items():
                        face_dir_dict[k] = v
            self.dataset.load_face_direction_feature(face_dir_dict)
            
        self.dataset.data = [d for d in self.dataset.data if not (d.pid == 12 and (d.place == "B" or d.place == "C"))]
    
    def load_conf_val(self, config, key):
        assert config is not None 
        assert key in config
        self.__dict__[key] = config[key]
 
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
            if ipath.pid in test_ids:
                ipath.type = 'test'
            elif ipath.pid in val_ids:
                ipath.type = 'validation'
            else:
                ipath.type = 'train'
        if self.bulking:
            ipaths = bulk_train_data(ipaths)
        
        if self.nonlocked_rate:
            ipaths = balance_train_data(ipaths, self.nonlocked_rate)

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

          
