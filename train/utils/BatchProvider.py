# -*- coding: utf-8 -*-

import os, sys
import random

import numpy as np
from PIL import Image, ImageOps

    
class BatchProvider():
    def __init__(self, train_paths, test_paths, batch_size, block_size, img_size, face_dir_dict=None):
        self.train_paths = train_paths
        self.test_paths = test_paths
        self.train_path_pool = None
        self.test_path_pool = None
        self.batch_size = batch_size
        self.block_size = block_size
        self.img_size = img_size
        #  batch_queue 
        self.train_block = []
        self.test_block = []
        
        if face_dir_dict:
            self.use_face_dir_feature = True
            self.face_dir_dict = face_direction_dict
        else:
            self.use_face_dir_feature = False
   
    def init(self, ran=True):
        if ran:
            self.train_path_pool = random.sample(self.train_paths, len(self.train_paths))
            self.test_path_pool = random.sample(self.test_paths, len(self.test_paths))
    
    def load_batch(self, paths, return_paths=False):
        xs = []
        ts = []
        
        if self.use_face_dir_feature:
            fs = []
        
        for path in paths:
            try:
                img = Image.open(path.path).convert('L') ## Gray->L, RGB->RGB
                img = ImageOps.equalize(img)
            except:
                print("Can't load %s." % path.path)
                continue
            
            img = img.resize((self.img_size[0], self.img_size[1]))
            
            if path.mirror:
                img = ImageOps.mirror(img)
            
            x = np.array(img, dtype=np.float32)
            x = x / 255.0 ## Normalize [0, 255] -> [0, 1]
            x = x.reshape(1, self.img_size[0], self.img_size[1]) ## Reshape image to input shape of CNN
            
            t = np.array(int(path.locked), dtype=np.int32)
            if self.use_face_dir_feature:
                f_list = self.face_dir_dict[path.img_name]
                f_list = [f / 255.0 for f in f_list]
                fs.append(np.array(f_list, dtype=np.float32))
            xs.append(x)
            ts.append(t)
        
        xs = np.array(xs).astype(np.float32)
        ts = np.array(ts).astype(np.int32)
        if return_paths:
            if self.use_face_dir_feature:
                return ((xs, fs), ts), paths
            else:
                return (xs, ts), paths
        else:
            if self.use_face_dir_feature:
                return (xs, fs), ts
            else:
                return xs, ts
    
    
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
        