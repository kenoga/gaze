# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image, ImageOps

# 識別器の学習に使う、入力と出力を読み込むためのクラス
class DataLoader(object):
    def __init__(self):
        pass

    def load(self):
        pass


class OmniDataLoader(DataLoader):
    def __init__(self, img_size):
        self.img_size = img_size
        pass

    def load(self, path):
        img = Image.open(path.path).convert('L') ## Gray->L, RGB->RGB
        img = ImageOps.equalize(img)

        img = img.resize((self.img_size[0], self.img_size[1]))

        if path.mirror:
            img = ImageOps.mirror(img)

        x = np.array(img, dtype=np.float32)
        x = x / 255.0 ## Normalize [0, 255] -> [0, 1]
        x = x.reshape(1, self.img_size[0], self.img_size[1]) ## Reshape image to input shape of CNN

        t = np.array(int(path.locked), dtype=np.int32)

        return x, t

class OmniWithFaceFeatureDataLoader(OmniDataLoader):
    def __init__(self, img_size):
        super(OmniWithFaceFeatureDataLoader, self).__init__(img_size)
        pass

    def load(self, path):
        x, t = super(OmniWithFaceFeatureLoader, self).load(path)

        f_list_nest = path.face_direction
        if path.mirror:
            # xを反転 (positionはx, yの順になっている
            f_list_nest = [[1-position[0], position[1]]for position in f_list_nest]
        f_list = [e / 255.0 for position in f_list_nest for e in position]
        f = np.array(f_list, dtype=np.float32)
        assert len(f) == 136
        return x, f, t
