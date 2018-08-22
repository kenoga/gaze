# -*- coding: utf-8 -*-

import random
import numpy as np
from PIL import Image, ImageOps

    
class BatchProvider():
    def __init__(self, train_paths, validation_paths, test_paths, batch_size, block_size, img_size, face_dir_dict):
        self.train_paths = train_paths
        self.validation_paths = validation_paths
        self.test_paths = test_paths
        self.train_path_pool = None
        self.validation_path_pool = None
        self.test_path_pool = None
        self.batch_size = batch_size
        self.block_size = block_size
        self.img_size = img_size
        #  batch_queue 
        self.train_block = []
        self.validation_block =[]
        self.test_block = []
        self.face_dir_dict = face_dir_dict  
 
    def init(self, ran=True):
        if ran:
            self.train_path_pool = random.sample(self.train_paths, len(self.train_paths))
            self.validation_path_pool = random.sample(self.validation_paths, len(self.validation_paths))
            self.test_path_pool = random.sample(self.test_paths, len(self.test_paths))
    
    def load_batch(self, paths, return_paths=False):
        xs = []
        ts = []
        
        if self.face_dir_dict:
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
            if self.face_dir_dict:
                f_list_nest = self.path.face_direction
                if path.mirror:
                    # xを反転 (positionはx, yの順になっている
                    f_list_nest = [[1-position[0], position[1]]for position in f_list_nest]
                f_list = [e / 255.0 for position in f_list_nest for e in position]
                assert len(f_list) == 136
                          
                fs.append(np.array(f_list, dtype=np.float32))
            xs.append(x)
            ts.append(t)
        
        xs = np.array(xs).astype(np.float32)
        ts = np.array(ts).astype(np.int32)
        if self.face_dir_dict:
            fs = np.array(fs).astype(np.float32)
        if return_paths:
            if self.face_dir_dict:
                return ((xs, fs), ts), paths
            else:
                return (xs, ts), paths
        else:
            if self.face_dir_dict:
                assert len(xs) == len(fs) == len(ts)
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
        
        
    def get_batch(self, dtype='train', paths=False):
        assert dtype in {'train', 'validation', 'test'}
        if self.train_path_pool is None or self.validation_path_pool is None or self.test_path_pool is None:
            print("Please initialize.")
            return None
        
        if dtype == 'train':
            path_pool = self.train_path_pool
            block = self.train_block
        elif dtype == 'validation':
            path_pool = self.validation_path_pool
            block = self.validation_block
        else:
            path_pool = self.test_path_pool
            block = self.test_block

        # blockが空だったらメモリに読み込む
        if len(block) == 0:
            #  全てのbatchをすでに吐き出していたらNoneを返す
            if len(path_pool) == 0:
                return None
#             print("Block loading now...")
            block.extend(self.load_block(path_pool, paths))
#             print("Block loading completed...")

        return block.pop()
        
