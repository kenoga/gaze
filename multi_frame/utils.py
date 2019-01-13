import os
import pympi
import numpy as np
import pickle
import glob
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def frame_num(img_dir, movie_id):
    return len(glob.glob(os.path.join(img_dir, movie_id, "*.png")))

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

def visualize(labels, predictions, frame_num, title, target=None, fig_path=None, big=True, start_t=0, second_as_xlabel=True):
    x = [i for i in range(1, frame_num+1)]
    if target:
        x = x[target[0]:target[1]]
        labels = labels[target[0]:target[1]]
        predictions = predictions[target[0]:target[1]]
    if big:
        figsize=(320, 10)
    else:
        figsize=(40,5)
    fig, axes = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=16)
    axes.plot(x, labels, linewidth=2, color="red")
    axes.plot(x, predictions, linewidth=0.3, color="#C3DEF1")
    plt.fill_between(x, predictions, 0, color="#C3DEF1")
    axes.set_ylim([0, 1.1])
    axes.set_ylabel("locked (1) or non-locked (0)")
    axes.set_xlabel("time (s)")
    axes.tick_params(direction='out', length=6, which='major')
    axes.tick_params(direction='out', length=3, which='minor')
    # set xticks at 1s (big) or 20s (not big)
    if big:
        ml = MultipleLocator(2)
        ml.MAXTICKS = 5000
        axes.set_xticks([i for i in range(0, frame_num+1, 20)])
        if second_as_xlabel:
            axes.set_xticklabels([i/20 for i in range(0, frame_num+1, 20)])
        axes.xaxis.set_minor_locator(ml)
    else:
        ml = MultipleLocator(20)
        ml.MAXTICKS = 5000
        axes.set_xticks([i for i in range(0, frame_num+1, 200)])
        if second_as_xlabel:
            axes.set_xticklabels([i/20+start_t for i in range(0, frame_num+1, 200)])
        axes.xaxis.set_minor_locator(ml)
        
    if fig_path:
        plt.savefig(os.path.join(fig_path))

def get_all_dataset_ids(img_dir, meta_dir):
    movie_ids = sorted([dname for dname in os.listdir(img_dir) if dname[0] != '.'])
    movie_ids = [(movie_id.split("_")[0], movie_id.split("_")[1]) for movie_id in movie_ids]
    dataset_ids = []
    for group_id, session_id in movie_ids:
        meta_path = os.path.join(meta_dir, "%s.json" % group_id)
        meta_info = json.load(open(meta_path))
        for seat_id in sorted(meta_info.keys()):
            dataset_ids.append((group_id, session_id, seat_id))
    return dataset_ids
