# -*- coding: utf-8 -*-

import os
import json
import cv2
import glob
from PIL import Image
from config import DS_ROOT
from IPython.display import Image, display

def imshow(img, format='.png'):
    decoded_bytes = cv2.imencode(format, img)[1].tobytes()
    display(Image(data=decoded_bytes))   

def imshow_by_path(path, format='.png'):
    img = cv2.imread(path)
    imshow(img)

def imshow_by_name(name, format='.png', type='aligned_face'):
    pid = name.split("_")[0]
    path = os.path.join(DS_ROOT, type, pid, name)
    imshow_by_path(path)
        
# person_id -> list of filenameのdictを返す
def get_image_paths(img_type):
    paths_dict = {}
    dsdir = os.path.join(DS_ROOT, img_type)
    subs = get_subdirs(img_type)
    
    for sub in subs:
        fpaths = [path for path in glob.glob(os.path.join(dsdir, sub, '*.jpg'))]
        paths_dict[sub] = fpaths        
    return paths_dict
    

# 指定されたディレクトリの子ディレクトリを返す
def get_subdirs(img_type='face_image'):
    return sorted([subdir for subdir in os.listdir(os.path.join(DS_ROOT, img_type)) if subdir[0] != '.'])


# person_id -> bb_and_landmarks dictのdictを返す
def get_bb_and_landmarks_dict():
    result = {}
    json_path = os.path.join(DS_ROOT, 'face_bb_and_landmarks', 'all.json')
    with open(json_path, 'r') as fr:
            return json.load(fr)


def get_face_direction_feature_dict():
    result = {}
    dir_path = os.path.join(DS_ROOT, 'face_direction_feature')
    jsons = glob.glob(os.path.join(dir_path, '*.json'))
    for json_file in jsons:
        json_path = os.path.join(DS_ROOT, json_file)
        with open(json_path, 'r') as fr:
            person_id = os.path.basename(json_file).split('.')[0]
            result[person_id] = json.load(fr)
    return result
