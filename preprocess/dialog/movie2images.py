# -*- coding: utf-8 -*-
import os, sys
import subprocess

import glob
sys.path.append("..")
sys.path.append("../..")
import dataset_utils.utils as utils
import dataset_utils.config as config


src_dir = os.path.abspath("/mnt/aoni02/nogawa/gaze/data/omni_dialog/real/movie")
movie_files = ["03_02.MP4"]
# movie_files = ["02_02.MP4"]
tgt_dir =  os.path.abspath("/mnt/aoni02/nogawa/gaze/data/omni_dialog/real/image")

for movie_file in movie_files:
    movie_id = movie_file.split(".")[0]
    movie_path = os.path.join(src_dir, movie_file)
    
    tgt_sub_dir = os.path.join(tgt_dir, movie_id)
    if not os.path.exists(tgt_sub_dir):
        os.mkdir(tgt_sub_dir)
    tgt_path = os.path.join(tgt_sub_dir, "%08d.png")
    
    command = "ffmpeg -i {0} -r 20 -q:v 1 -f image2 {1}".format(movie_path, tgt_path).split(" ")
    print(command)
    
    print(' '.join(command))
    subprocess.call(command)
utils.notify("movie2images done!!!")
