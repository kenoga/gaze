# -*- coding: utf-8 -*-
import os, sys, glob, json
import pickle
from gaze_detector import GazeDetector
sys.path.append(os.path.abspath(".."))
import dataset_utils.utils as utils
import dataset_utils.config as config
from train.model.cnn import CNN

def load_meta_info(meta_dir, movie_id):
    meta_path = os.path.join(meta_dir, "%s.json" % movie_id.split("_")[0])
    with open(meta_path, "r") as fr:
        meta_info = json.load(fr)
    return meta_info

def load_trained_gaze_detector_for_each_pid(model_dir, pid):
    ## 最善のモデルを選択する　or モデル生成時に最善のモデルしか保存しない
    model_path = os.path.join(model_dir, "for_pid_%02d" % pid, "for_pid_%02d_00.npz" % pid)
    model = CNN()
    return GazeDetector(model, model_path)

def load_trained_gaze_detector(model_dir):
    model_path = os.path.join(model_dir, "for_multi_frame_test_group_id_5", "for_multi_frame_test_group_id_5_02.npz")
    model = CNN()
    return GazeDetector(model, model_path)


base_dir = "../data/omni_dialog/real/"
src_dir = os.path.join(base_dir, "both_eyes_from_aligned_face")
tgt_dir = os.path.join(base_dir, "detection_result")
meta_dir = os.path.join(base_dir, "meta_info")
model_dir = "../train/trained"

movie_ids = ["01_01", "01_02", "02_01", "02_02", "03_01", "03_02", "04_01", "04_02", "05_01", "05_02"]

for movie_id in movie_ids:
    meta_info = load_meta_info(meta_dir, movie_id)
    seats = sorted(meta_info.keys())
    for seat in seats:
        gaze_detector = load_trained_gaze_detector(model_dir)
        results = {}
        src_sub_dir = os.path.join(src_dir, movie_id)
        img_paths = sorted(glob.glob(os.path.join(src_sub_dir, "*_%s.png" % seat)))
        for img_path in img_paths:
            result = gaze_detector.detect(img_path)
            fc1 = gaze_detector.get_fc1(img_path)
            fc2 = gaze_detector.get_fc2(img_path)
            img_name = os.path.basename(img_path)
            results[img_name] = {"score": result, "fc1": fc1, "fc2": fc2}
            print("%s -> %s" % (img_name, str(result)))
        
        with open(os.path.join(tgt_dir, "%s_%s.pickle" % (movie_id, seat)), "w") as fw:
            pickle.dump(results, fw)

