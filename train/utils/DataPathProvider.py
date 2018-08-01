# -*- coding: utf-8 -*-

import os, sys
import random
import glob
import json
import copy

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

def load_fujikawa_data(dtype="train"):
    images = []
    if dtype == 'train':
        ds_path = '../data/fujikawa/train'
    else:
        ds_path = '../data/fujikawa/test'
    locked_paths = glob.glob(os.path.join(ds_path, 'locked', '*.png'))
    nonlocked_paths = glob.glob(os.path.join(ds_path, 'nonlocked', '*.png'))
    assert len(locked_paths) > 1
    assert len(nonlocked_paths) > 1
    for path in locked_paths:
        fip = FujikawaImagePath(path)
        fip.locked = True
        images.append(fip)
    for path in nonlocked_paths:
        fip = FujikawaImagePath(path)
        fip.locked = False
        images.append(fip)
    return images

def load_imgpaths(dataset_dir, validation_ids, test_ids, locked_targets, ignored_targets, places, \
                  train_ids=None, \
                  annotation_dict=None,
                  nonlocked_rate=None,
                  bulking=False,
                  skip_num=0):
    assert os.path.exists(dataset_dir)
    assert nonlocked_rate >= 1 or nonlocked_rate is None or nonlocked_rate == False
    assert skip_num >= 0

    imgpaths = sorted(glob.glob(os.path.join(dataset_dir, '*/*.jpg')))
    imgpaths = imgpaths[::skip_num+1]
    imgs = [ImagePath(path) for path in imgpaths]
    
    if train_ids:
        imgs = [img for img in imgs if img.pid in train_ids or img.pid in test_ids or img.pid in validation_ids]

    # delete images annotated as noise data.
    if annotation_dict:
        imgs = [img for img in imgs if img.img_name not in annotation_dict or annotation_dict[img.img_name] not in {'closed-eyes', 'other'}]

    # delete images those targets should be ignored
    imgs = [img for img in imgs if img.target not in ignored_targets]
   
    for img in imgs:
        img.locked = True if img.target in locked_targets else False
    
    for img in imgs:
        if img.pid in test_ids:
            img.type = 'test'
        elif img.pid in validation_ids:
            img.type = 'validation'
        else:
            img.type = 'train'
    
    if bulking:
        imgs = bulk_train_data(imgs)
    
    if nonlocked_rate:
        imgs = balance_train_data(imgs, nonlocked_rate)

    train = []
    validation = []
    test = []
    
    for img in imgs:
        if img.type == 'train':
            train.append(img)
        elif img.type == 'validation':
            validation.append(img)
        else:
            test.append(img)

    return train, validation, test

class DataPathProvider():
    def __init__(self, dataset_dir, validation_ids, test_ids, 
                 locked_targets=[0,1,2,31,32, 40, 50], 
                 ignored_targets={}, 
                 places={"A","B","C","D"}, 
                 train_ids=None, 
                 bulking=False,
                 nonlocked_rate=1,
                 fujikawa_dataset=None,
                 annotation_path=None, 
                 face_dir_dict=None,
                 skip_num=0):
        assert fujikawa_dataset in {False, "train", "only_test", "only"}
        # fujikawa_dataset
        # False -> train: omni, test: omni
        # train -> train: omni + fujikawa, test: omni
        # only_test -> train: omni, test: fujikawa
        # only -> train: fujikawa, test: fujikawa
        
        annotation_dict = None
        if annotation_path:
            with open(annotation_path, 'r') as fr:
                annotation_dict = json.load(fr)
        
        if face_dir_dict:
            self.use_face_dir_feature = True
            self.face_dir_dict = face_direction_dict
        else:
            self.use_face_dir_feature = False
            
        self.train_paths, self.validation_paths, self.test_paths = load_imgpaths(dataset_dir, \
                                                          validation_ids, \
                                                          test_ids, \
                                                          locked_targets, \
                                                          ignored_targets, \
                                                          places, train_ids=train_ids, \
                                                          annotation_dict=annotation_dict,
                                                          nonlocked_rate=nonlocked_rate,
                                                          bulking=bulking,
                                                          skip_num=skip_num)
        # TODO: test画像がダメ
#         if fujikawa_dataset == "train":
#             fujikawa_data = load_fujikawa_data(dtype="train")
#             self.train_paths.extend(fujikawa_data)
#         elif fujikawa_dataset == "only_test":
#             fujikawa_data = load_fujikawa_data(dtype="train")
#             self.test_paths = fujikawa_data
#         elif fujikawa_dataset == "only":
#             fujikawa_train = load_fujikawa_data(dtype="train")
#             fujikawa_test = load_fujikawa_data(dtype="test")
#             self.train_paths = fujikawa_train
#             self.test_paths = fujikawa_test
                                    
    def get_paths(self):
        return self.train_paths, self.validation_paths, self.test_paths
    
    def report(self):
        print(' '.join(['-' * 25, 'dataset', '-' * 25]))
        print("train locked size: %d" % len([0 for path in self.train_paths if path.locked == True]))
        print("train nonlocked size: %d" % len([0 for path in self.train_paths if path.locked == False]))
        print("validation locked size: %d" % len([0 for path in self.validation_paths if path.locked == True]))
        print("validation nonlocked size: %d" % len([0 for path in self.validation_paths if path.locked == False]))
        print("test locked size: %d" % len([0 for path in self.test_paths if path.locked == True]))
        print("test nonlocked size: %d" % len([0 for path in self.test_paths if path.locked == False]))
        
        all_paths = []
        all_paths.extend(self.train_paths)
        all_paths.extend(self.validation_paths)
        all_paths.extend(self.test_paths)
        print("all locked sizes: %d" % len([0 for path in all_paths if path.locked == True]))
        print("all unlocked sizes: %d" % len([0 for path in all_paths if path.locked == False]))
          