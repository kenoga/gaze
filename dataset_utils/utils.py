# -*- coding: utf-8 -*-

import os
import json
import requests
import cv2
import glob
import csv
from PIL import Image
from config import DS_ROOT
from IPython.display import Image, display
try:
    from openface.align_dlib import TEMPLATE
except ImportError:
    pass

def notify(text):
    requests.post('https://hooks.slack.com/services/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', data = json.dumps({
        'text': text, # 投稿するテキスト
        'username': u'notification', # 投稿のユーザー名
        'icon_emoji': u':ghost:', # 投稿のプロフィール画像に入れる絵文字
        'link_names': 1, # メンションを有効にする
    }))
    
def get_place_region_in_panorama(img, place):
    height, width = tuple(img.shape[:2])
    if place == "A":
        return img[height//2:, width//2+2200:width//2+2880]
    elif place == "B":
        return img[height//2:, width//2+70:width//2+750]
    elif place == "C":
        return img[height//2:, width//2+1750:width//2+2200]
    elif place == "D":
        return img[height//2:, width//2+750:width//2+1200]
    else:
        raise Exception
    
def imwrite(path, img):
    cv2.imwrite(path, img)

def imread(path):
    return cv2.imread(path)

def imshow(img, format='.png'):
    decoded_bytes = cv2.imencode(format, img)[1].tobytes()
    display(Image(data=decoded_bytes))   

def imshow_by_path(path, format='.png'):
    img = cv2.imread(path)
    imshow(img)

def imshow_by_name(name, format='.png', type='aligned_face2'):
    pid = name.split("_")[0]
    path = os.path.join(DS_ROOT, type, pid, name)
    imshow_by_path(path)
        
# person_id -> list of filenameのdictを返す
def get_image_paths(img_type, dataset_path=DS_ROOT, format='jpg'):
    paths_dict = {}
    dsdir = os.path.join(dataset_path, img_type)
    subs = get_subdirs(img_type, dataset_path=dataset_path)
    
    for sub in subs:
        fpaths = [path for path in glob.glob(os.path.join(dsdir, sub, '*.%s' % format))]
        paths_dict[sub] = fpaths        
    return paths_dict
    

# 指定されたディレクトリの子ディレクトリを返す
def get_subdirs(img_type='face_image', dataset_path=DS_ROOT):
    return sorted([subdir for subdir in os.listdir(os.path.join(dataset_path, img_type)) if subdir[0] != '.'])


# person_id -> bb_and_landmarks dictのdictを返す
def get_bb_and_landmarks_dict():
    result = {}
    dir_path = os.path.join(DS_ROOT, 'face_bb_and_landmarks')
    jsons = [path for path in glob.glob(os.path.join(dir_path, '*.json')) if "all.json" not in path]
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

def load_triangles():
    triangles = []
    triangle_csv = os.path.join(os.path.dirname(__file__), "triangle.csv")
    with open(triangle_csv) as fr:
        csvreader = csv.reader(fr)
        for triangle in csvreader:
            triangles.append([int(p) for p in triangle])
    return triangles

def extract_eyes_region_from_aligned_face(img, margin=0):
    assert img is not None
    assert img.shape[0] == img.shape[1]
    size = img.shape[0]
    top = int(TEMPLATE[37][1]*size) - margin
    bottom = int(TEMPLATE[41][1]*size) + margin
    right = int(TEMPLATE[36][0]*size) - margin
    left = int(TEMPLATE[45][0]*size) + margin
    eyes = img[top: bottom, right: left]
    return eyes

def extract_eye_regions_from_aligned_face(img, margin=0):
    assert img is not None
    assert img.shape[0] == img.shape[1]
    size = img.shape[0]
    
    eye_width = int(TEMPLATE[39][0]*size) - int(TEMPLATE[36][0]*size)
    eye_height = int(TEMPLATE[41][1]*size) - int(TEMPLATE[37][1]*size)
    
    right_top = int(TEMPLATE[37][1]*size)
    right_right = int(TEMPLATE[36][0]*size)
    right_eye = img[right_top-margin: right_top+eye_height+margin, right_right-margin: right_right+eye_width+margin]
   
    left_top = int(TEMPLATE[44][1]*size)
    left_right = int(TEMPLATE[42][0]*size)
    left_eye = img[left_top-margin: left_top+eye_height+margin, left_right-margin: left_right+eye_width+margin]
    return left_eye, right_eye
    
def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def eqhist(img):
    return cv2.equalizeHist(img)
