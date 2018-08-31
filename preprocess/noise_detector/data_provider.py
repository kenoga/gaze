# -*- coding: utf-8 -*- 

import os, sys
import json

module_path = os.path.abspath(os.path.join("..", ".."))
sys.path.append(module_path)
from dataset_utils.config import DS_ROOT
from dataset_utils.utils import get_image_paths
from dataset_utils.DataInitiator import OmniDataInitiator


ANT_PATH = "./flmk_annotation.json"
DATA_DIR = os.path.join(DS_ROOT, "aligned_68")
data_initiator = OmniDataInitiator()


# ノイズ検出器の学習，検証，テストに必要なデータを作成して返す

with open(ANT_PATH, "r") as fr:
    ant = json.load(fr)
    
img_paths = get_image_paths("aligned_68")


pos_data = []
neg_data = []
for data in [data_initiator.init(path) for path in img_paths]:
    # annotation対象のデータの条件に当てはまるものだけをデータセットに加える
    if data.id % 5 != 0 or data.id > 50:
        continue
        
    if data.name not in ant or ant[data.name] == "ok":
        pos_data.append(data)
    elif ant[data.name] == "big_gap":
        neg_data.append(data)
        
print(pos_data)

print(neg_data)
        
    