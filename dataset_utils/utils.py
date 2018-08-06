# -*- coding: utf-8 -*-

import os
import json
import cv2
import glob
from config import DS_ROOT

def imshow(img, format='.png'):
    decoded_bytes = cv2.imencode(format, img)[1].tobytes()
    display(Image(data=decoded_bytes))   

        
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
    dir_path = os.path.join(DS_ROOT, 'face_bb_and_landmarks')
    jsons = glob.glob(os.path.join(dir_path, '*.json'))
    for json_file in jsons:
        json_path = os.path.join(DS_ROOT, json_file)
        with open(json_path, 'r') as fr:
            person_id = os.path.basename(json_file).split('.')[0]
            result[person_id] = json.load(fr)
    return result


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
