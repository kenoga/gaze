# -*- coding: utf-8 -*-
import os, sys
import glob
import json
import numpy as np
import pickle
from collections import defaultdict

import pympi

sys.path.append("..")
sys.path.append("../..")
import dataset_utils.utils as utils
import dataset_utils.config as config

FC1_DIM = 256
FC2_DIM = 128
FC3_DIM = 1

def ms2frameid(ms):
    frame_ms = 50
    ms_rounded = ms - (ms % 50)
    return (ms_rounded / frame_ms)

def load_ts(movie_id, seat_id, ant_dir, frame_num):
    fname = "%s_%s.eaf" % (movie_id, seat_id)
    path = os.path.join(ant_dir, fname)
    eaf = pympi.Elan.Eaf(path)
    aid2tid = eaf.tiers["default"][0]
    tid2ms = eaf.timeslots
    slots = []
    for aid in sorted(aid2tid.keys(), key=lambda key: int(key[1:])):
        tid_start = aid2tid[aid][0]
        tid_end = aid2tid[aid][1]
        ms_start = tid2ms[tid_start]
        ms_end = tid2ms[tid_end]
        start = ms2frameid(ms_start)
        end = ms2frameid(ms_end)
        slots.append((start, end))
    labels = [0 for _ in range(frame_num)]
    for start, end in slots:
        try:
            for i in range(start, end):
                labels[i] = 1
        except Exception as e:
            print("error")
            print(start, end, frame_num)
            raise e
    return labels
    
    
def load_xs(movie_id, seat_id, detection_result_dir, frame_num, x_type):
    labels = [np.array([0], dtype=np.float32) for _ in range(frame_num)]
    fname = "%s_%s.pickle" % (movie_id, seat_id)
    path = os.path.join(detection_result_dir, fname)
    with open(path, "r") as fr:
        results = pickle.load(fr)

    if x_type == "fc1":
        dim = 256
    elif x_type == "fc2":
        dim = 128
    else:
        dim = 1
    xs = [np.zeros((dim), dtype=np.float32) for _ in range(frame_num)]
    for fname, result in sorted(results.items()):
        if dim==1:
            output = np.array([result[x_type][1]], dtype=np.float32)
        else:
            output = np.array(result[x_type], dtype=np.float32)
        fid = int(fname.split("_")[0])
        xs[fid-1] = output
    return xs


def make_dataset(movie_ids, x_type):
    base_path = "../data/omni_dialog/real/"
    prediction_dir = os.path.join(base_path, "detection_result")
    label_dir = os.path.join(base_path, "ant")
    img_dir = os.path.join(base_path, "image")
    meta_dir = os.path.join(base_path, "meta_info")
    
    dataset = defaultdict(defaultdict)
    for movie_id in movie_ids:
        dialog_id = int(movie_id.split("_")[0])
        session_id = int(movie_id.split("_")[1])
        dataset[dialog_id][session_id] = {}
        meta_path = os.path.join(meta_dir, "%s.json" % (movie_id.split("_")[0]))
        with open(meta_path, "r") as fr:
            meta_info = json.load(fr)
        seats = meta_info.keys()
        frame_num = len(glob.glob(os.path.join(img_dir, movie_id, "*.png")))
        for seat_id in seats:
            xs = load_xs(movie_id, seat_id, prediction_dir, frame_num, x_type)
            ts = load_ts(movie_id, seat_id, label_dir, frame_num)
            dataset[dialog_id][session_id][seat_id] = (xs, ts)
    return dataset
        
    
if __name__ == "__main__":
    movie_ids = ["01_01", "01_02", "02_01", "02_02", "03_01", "03_02", "04_01", "04_02", "05_01", "05_02"]
#     x_type = "score"
#     dataset = make_dataset(movie_ids, x_type)
#     with open("./dataset/dataset_fc3.pickle", "w") as fw:
#         pickle.dump(dataset, fw)
        
#     x_type = "fc2"
#     dataset = make_dataset(movie_ids, x_type)
#     with open("./dataset/dataset_fc2.pickle", "w") as fw:
#         pickle.dump(dataset, fw)

    x_type = "fc1"
    dataset = make_dataset(movie_ids, x_type)
        with open("./dataset/dataset_fc1.pickle", "w") as fw:
        pickle.dump(dataset, fw)
    
    