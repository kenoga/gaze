# -*- coding: utf-8 -*-

import os
import sys
import json
import cv2

module_path = os.path.abspath("..")
if module_path not in sys.path:
    sys.path.append(module_path)
    
from utils.FaceAligner import FaceAligner
from dataset_utils.config import DS_ROOT, DS_KATAYAMA
from dataset_utils.utils import get_image_paths


def main(src_paths, dst_dir, json_dir):
    # subdirごとに結果のjsonファイルを保存する
    json_fname = "all.json"
    img_path_dict = src_paths
    face_root_dir = dst_dir
    json_dir = json_dir
    aligner = FaceAligner()
    
    data_size = sum([len(img_path_dict[subdir]) for subdir in img_path_dict.keys()])
    print("data_size: %d" % data_size)
    count = 0

    if not os.path.exists(json_dir):
        os.mkdir(json_dir)

    results = {}    
    for sub_dir in sorted(img_path_dict.keys()):
        img_paths = img_path_dict[sub_dir]
        face_dir = os.path.join(face_root_dir, sub_dir)    

        if not os.path.exists(face_dir):
            os.makedirs(face_dir)

        for img_path in img_paths:
            count += 1
            sys.stdout.write('\r'+"person: %d / %d, all: %d / %d" % (count, len(img_paths), count, data_size))
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)
            bb = aligner.bb(img=img)

            if bb is not None:
                landmarks = aligner.landmarks(img=img, bb=bb)
                # save face img
                face_img = img[bb.top():bb.bottom(), bb.left():bb.right()]
                cv2.imwrite(os.path.join(face_dir, img_name), face_img)
                results[img_name] = {}
                results[img_name]['bb'] = [bb.left(), bb.top(), bb.right(), bb.bottom()]
                results[img_name]["landmarks"] = landmarks

    with open(os.path.join(json_dir, json_fname), "w") as fr:
        json.dump(results, fr, sort_keys=True)

src_paths = get_image_paths('image', dataset_path=DS_KATAYAMA, format='png')
dst_dir =  os.path.join(DS_KATAYAMA, "face")
json_dir = os.path.join(DS_KATAYAMA, "face_bb_and_landmarks")
main(src_paths, dst_dir, json_dir)