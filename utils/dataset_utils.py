# -*- coding: utf-8 -*-

import os
import sys
import glob
import json
import cv2
from openface import AlignDlib

from IPython.display import display, Image

DS_TYPES = {'movie', 'image', 'transformed_image', 'both_eyes', 'face_image', 'aligned_face'}
DS_ROOT_DIR = '/root/gaze/data/kobas-omni-eyecontact'


def imshow(img, format='.png'):
    decoded_bytes = cv2.imencode(format, img)[1].tobytes()
    display(Image(data=decoded_bytes))   


# src_dirの子ディレクトリ(sub_dir)の中に含まれる全ての画像(img)に対してfuncを適用してdst_dir/sub_dir/new_imgとして保存する
def apply_func_against_srcdir_and_save_to_dstdir(src_dir, dst_dir, func):
    assert os.path.exists(src_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    paths = glob.glob(os.path.join(src_dir, '*', '*.jpg'))
    print(len(paths))
    img_num = len(paths)
    count = 1
    for path in paths:
        sys.stdout.write('\r'+"%d / %d" % (count, img_num))
        count += 1
        
        img = cv2.imread(path)
        # funcは新しいimgまたはNoneを返す
        new_img = func(img)
        if new_img is None:
            continue
        # サブディレクトリ名の取得
        sub_dir = os.path.basename(os.path.dirname(path))
        fname = os.path.basename(path)
        dst_sub_dir = os.path.join(dst_dir, sub_dir)
        if not os.path.exists(dst_sub_dir):
            os.mkdir(dst_sub_dir)
        cv2.imwrite(os.path.join(dst_sub_dir, fname), new_img)
        

    
# person_id -> list of filenameのdictを返す
def get_image_paths(img_type):
    paths_dict = {}
    dsdir = os.path.join(DS_ROOT_DIR, img_type)
    subs = get_subdirs(img_type)
    
    for sub in subs:
        fpaths = [path for path in glob.glob(os.path.join(dsdir, sub, '*.jpg'))]
        paths_dict[sub] = fpaths        
    return paths_dict


# 指定されたディレクトリの子ディレクトリを返す
def get_subdirs(img_type='face_image'):
    return sorted([subdir for subdir in os.listdir(os.path.join(DS_ROOT_DIR, img_type)) if subdir[0] != '.'])



# person_id -> bb_and_landmarks dictのdictを返す
def get_bb_and_landmarks_dict():
    result = {}
    dir_path = os.path.join(DS_ROOT_DIR, 'face_bb_and_landmarks')
    jsons = glob.glob(os.path.join(dir_path, '*.json'))
    for json_file in jsons:
        json_path = os.path.join(DS_ROOT_DIR, json_file)
        with open(json_path, 'r') as fr:
            person_id = os.path.basename(json_file).split('.')[0]
            result[person_id] = json.load(fr)
    return result
            
class FaceAligner():
    def __init__(self):
        DLIB_MODEL_PATH = '/root/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
        self.aligner = AlignDlib(DLIB_MODEL_PATH)
        
    def align(self, img_path=None, img=None, bb=None, landmarks=None, size=255):
        if img_path is None and img is None:
            raise RuntimeError("Please give a imgpath or a img.")
        if img_path:
            img = cv2.imread(img_path)
        return self.aligner.align(255, img, bb=bb, landmarks=landmarks)
    
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

 

                          

