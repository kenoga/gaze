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

        img = img.resize((self.img_size[1], self.img_size[0]))

        if path.mirror:
            img = ImageOps.mirror(img)

        x = np.array(img, dtype=np.float32)
        x = x / 255.0 ## Normalize [0, 255] -> [0, 1]
        x = x.reshape(1, self.img_size[0], self.img_size[1]) ## Reshape image to input shape of CNN

        t = np.array(int(path.locked), dtype=np.int32)

        return x, t


class OnlyFaceFeatureDataLoader(DataLoader):
    def __init__(self, img_size):
        self.img_size = img_size
        pass

    def load(self, path):
        t = np.array(int(path.locked), dtype=np.int32)

        f_list_nest = path.face_direction
        if path.mirror:
            # xを反転 (positionはx, yの順になっている
            f_list_nest = [[1-position[0], position[1]]for position in f_list_nest]
        f_list = [e / 255.0 for position in f_list_nest for e in position]
        f = np.array(f_list, dtype=np.float32)
        assert len(f) == 136

        return f, t



class OmniWithFaceFeatureDataLoader(OmniDataLoader):
    def __init__(self, img_size):
        super(OmniWithFaceFeatureDataLoader, self).__init__(img_size)
        pass

    def load(self, path):
        x, t = super(OmniWithFaceFeatureDataLoader, self).load(path)

        f_list_nest = path.face_direction
        if path.mirror:
            # xを反転 (positionはx, yの順になっている
            f_list_nest = [[1-position[0], position[1]]for position in f_list_nest]
        f_list = [e / 255.0 for position in f_list_nest for e in position]
        f = np.array(f_list, dtype=np.float32)
        assert len(f) == 136
        return x, f, t

class OmniWithFaceFeatureWithoutEyeRegionDataLoader(OmniDataLoader):
    def __init__(self, img_size):
        super(OmniWithFaceFeatureWithoutEyeRegionDataLoader, self).__init__(img_size)
        pass

    def load(self, path):
        eye_region = set([i for i in range(36, 46)])
        x, t = super(OmniWithFaceFeatureWithoutEyeRegionDataLoader, self).load(path)

        f_list_nest = []
        for i in range(len(path.face_direction)):
            if i not in eye_region:
                f_list_nest.append(path.face_direction[i])

        if path.mirror:
            # xを反転 (positionはx, yの順になっている
            f_list_nest = [[1-position[0], position[1]]for position in f_list_nest]
        f_list = [e / 255.0 for position in f_list_nest for e in position]
        f = np.array(f_list, dtype=np.float32)
        return x, f, t

class OmniWithFaceFeatureOnlyEyeRegionDataLoader(OmniDataLoader):
    def __init__(self, img_size):
        super(OmniWithFaceFeatureOnlyEyeRegionDataLoader, self).__init__(img_size)
        pass

    def load(self, path):
        eye_region = set([i for i in range(36, 46)])
        x, t = super(OmniWithFaceFeatureOnlyEyeRegionDataLoader, self).load(path)

        f_list_nest = []
        for i in range(len(path.face_direction)):
            if i in eye_region:
                f_list_nest.append(path.face_direction[i])

        if path.mirror:
            # xを反転 (positionはx, yの順になっている
            f_list_nest = [[1-position[0], position[1]]for position in f_list_nest]
        f_list = [e / 255.0 for position in f_list_nest for e in position]
        f = np.array(f_list, dtype=np.float32)
        return x, f, t


class OmniFaceFeaturePlaceDataLoader(OmniWithFaceFeatureDataLoader):
    def __init__(self, img_size):
        self.img_size = img_size
        pass

    def _place_to_place_id(self, place):
        place_map = {"A": 0, "B": 0, "C": 1, "D": 1}
        return place_map[place]

    def load(self, path):
        x, f, t = super(OmniFaceFeaturePlaceDataLoader, self).load(path)
        place_id = self._place_to_place_id(path.place)
        place_id = np.array([place_id], dtype=np.float32)
        return x, f, place_id, t


class OmniReversedEachEyeDataLoader(OmniDataLoader):
    def __init__(self, img_size):
        super(OmniReversedEachEyeDataLoader, self).__init__(img_size)
        pass

    def load(self, path):
        img = Image.open(path.path).convert('L') ## Gray->L, RGB->RGB
        img = ImageOps.equalize(img)

        img = img.resize((self.img_size[1], self.img_size[0]))

        if path.mirror:
            img = ImageOps.mirror(img)

        left_eye = img.crop((0, 0, int(self.img_size[1]/2), self.img_size[0]))
        right_eye = img.crop((int(self.img_size[1]/2), 0, self.img_size[1], self.img_size[0]))
        right_eye = ImageOps.mirror(right_eye) # 反転させる

        l = np.array(left_eye, dtype=np.float32) / 255.0
        r = np.array(right_eye, dtype=np.float32) / 255.0

        l = l.reshape(1, int(self.img_size[1]/2), self.img_size[0]) ## Reshape image to input shape of CNN
        r = r.reshape(1, int(self.img_size[1]/2), self.img_size[0]) ## Reshape image to input shape of CNN

        t = np.array(int(path.locked), dtype=np.int32)

        return l, r, t

class OmniEachEyeDataLoader(OmniDataLoader):
    def __init__(self, img_size):
        super(OmniEachEyeDataLoader, self).__init__(img_size)
        pass

    def load(self, path):
        img = Image.open(path.path).convert('L') ## Gray->L, RGB->RGB
        img = ImageOps.equalize(img)

        img = img.resize((self.img_size[1], self.img_size[0]))

        if path.mirror:
            img = ImageOps.mirror(img)

        left_eye = img.crop((0, 0, int(self.img_size[1]/2), self.img_size[0]))
        right_eye = img.crop((int(self.img_size[1]/2), 0, self.img_size[1], self.img_size[0]))

        l = np.array(left_eye, dtype=np.float32) / 255.0
        r = np.array(right_eye, dtype=np.float32) / 255.0

        l = l.reshape(1, int(self.img_size[1]/2), self.img_size[0]) ## Reshape image to input shape of CNN
        r = r.reshape(1, int(self.img_size[1]/2), self.img_size[0]) ## Reshape image to input shape of CNN

        t = np.array(int(path.locked), dtype=np.int32)

        return l, r, t

class OmniEachEyeWithFaceFeatureDataLoader(OmniDataLoader):
    def __init__(self, img_size):
        super(OmniEachEyeWithFaceFeatureDataLoader, self).__init__(img_size)
        pass

    def load(self, path):
        img = Image.open(path.path).convert('L') ## Gray->L, RGB->RGB
        img = ImageOps.equalize(img)

        img = img.resize((self.img_size[1], self.img_size[0]))

        if path.mirror:
            img = ImageOps.mirror(img)

        left_eye = img.crop((0, 0, int(self.img_size[1]/2), self.img_size[0]))
        right_eye = img.crop((int(self.img_size[1]/2), 0, self.img_size[1], self.img_size[0]))
        right_eye = ImageOps.mirror(right_eye) # 反転させる

        l = np.array(left_eye, dtype=np.float32) / 255.0
        r = np.array(right_eye, dtype=np.float32) / 255.0

        l = l.reshape(1, int(self.img_size[1]/2), self.img_size[0]) ## Reshape image to input shape of CNN
        r = r.reshape(1, int(self.img_size[1]/2), self.img_size[0]) ## Reshape image to input shape of CNN

        t = np.array(int(path.locked), dtype=np.int32)

        f_list_nest = path.face_direction
        if path.mirror:
            # xを反転 (positionはx, yの順になっている
            f_list_nest = [[1-position[0], position[1]]for position in f_list_nest]
        f_list = [e / 255.0 for position in f_list_nest for e in position]
        f = np.array(f_list, dtype=np.float32)
        assert len(f) == 136

        return l, r, f, t
