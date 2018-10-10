# -*- coding: utf-8 -*-

import random
import numpy as np
from PIL import Image, ImageOps


class BatchProvider():
    def __init__(self, data_loader, train_paths, validation_paths, test_paths, batch_size, block_size, img_size):
        self.data_loader = data_loader
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

    def init(self, ran=True):
        if ran:
            self.train_path_pool = random.sample(self.train_paths, len(self.train_paths))
            self.validation_path_pool = random.sample(self.validation_paths, len(self.validation_paths))
            self.test_path_pool = random.sample(self.test_paths, len(self.test_paths))


    def load_batch(self, paths, return_paths=False):
        # data_loaderの戻り値の数が設定によって変わる
        result = []

        for path in paths:
            data = self.data_loader.load(path)

            # 初期化処理：data_loaderの戻り値の個数だけ追加
            if not result:
                for _ in data:
                    result.append([])

            for i, d in enumerate(data):
                result[i].append(d)

        for i in range(len(result)):
            dtype = result[i][0].dtype
            result[i] = np.array(result[i], dtype=dtype)

        if return_paths:
            result.append(paths)

        return tuple(result)


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
