# -*- coding: utf-8 -*-

import cv2
import numpy as np
import openface
from openface import AlignDlib

class FaceAligner():
    def __init__(self), dlib_model_path='/root/openface/models/dlib/shape_predictor_68_face_landmarks.dat'):
        self.aligner = AlignDlib(dlib_model_path)
        
    def align(self, img=None, bb=None, landmarks=None, size=None, indices=None):
        assert img is not None
        assert bb is not None
        assert landmarks is not None
        assert size is not None
        assert indices is not None
        
        return self.aligner.align(size, img, bb=bb, landmarks=landmarks)
    
    def bb(self, img_path=None, img=None):
        if img_path is None and img is None:
            raise RuntimeError("Please give a imgpath or a img.")
        if img_path:
            img = cv2.imread(imgpath)
        return self.aligner.getLargestFaceBoundingBox(img)
    
    def landmarks(self, img_path=None, img=None, bb=None):
        if bb is None:
            raise RuntimeError("Please give a bb.")
        if img_path is None and img is None:
            raise RuntimeError("Please give a imgpath or a img.")
        if img_path:
            img = cv2.imread(imgpath)
        return self.aligner.findLandmarks(img, bb)
        
    # アフィン変換後の顔の基準点の位置を返す
    def get_landmarks_after_align(self, landmarks, img_size, indices=[39, 42, 57]):
        assert(type(landmarks) == list)
        assert(len(landmarks) == 68)
        assert(indices is not None)
        assert(type(indices) == list)
        assert(len(indices) == 3)
        indices = np.array(indices)
        landmarks = np.float32(landmarks)
        H = cv2.getAffineTransform(landmarks[indices], img_size * openface.align_dlib.MINMAX_TEMPLATE[indices])
        result = []
        for lm in landmarks:
            lm = np.hstack((lm, 1))
            pos = np.dot(H, lm)
            result.append(pos.tolist())
        return result