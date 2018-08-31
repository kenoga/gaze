# -*- coding: utf-8 -*- 

import os, sys
import json
import cv2

module_path = os.path.abspath(os.path.join("..", ".."))
sys.path.append(module_path)
from dataset_utils.config import DS_ROOT
from dataset_utils.utils import get_image_paths
from dataset_utils.Dataset import Dataset
from dataset_utils.utils import extract_eyes_region_from_aligned_face, gray, eqhist


ANT_PATH = "./flmk_annotation.json"
DATA_DIR = os.path.join(DS_ROOT, "aligned_with_68")

# ノイズ検出器の学習，検証，テストに必要なデータを作成して返す
noise_label = {"big_gap", "one_eye", "small_gap", "blink", "other"}


def get_data(data_dir, ant_path):
    dataset = Dataset(data_dir)
    
    with open(ANT_PATH, "r") as fr:
        ant = json.load(fr)
        
    result = []
    for data in dataset.data:
        # annotation対象のデータの条件に当てはまるものだけをデータセットに加える
        if data.glasses:
            continue
        if data.id % 5 != 0 or data.id > 50:
            continue
            
        if data.name not in ant or ant[data.name] not in noise_label:
            data.label = 0
            result.append(data)
        elif ant[data.name] in noise_label:
            data.label = 1
            result.append(data)
    
    for d in result:
        img = cv2.imread(d.path)
        d.img = eqhist(gray(extract_eyes_region_from_aligned_face(img)))
        d.x = d.img.flatten()
    return result