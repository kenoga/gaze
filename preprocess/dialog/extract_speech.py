# -*- coding: utf-8 -*-
import os, sys
import subprocess

import glob
sys.path.append("..")
sys.path.append("../..")
import dataset_utils.utils as utils
import dataset_utils.config as config


src_dir = os.path.abspath("/mnt/aoni02/nogawa/gaze/data/omni_dialog/real/movie")
movie_files = ["05_01.MP4", "05_02.MP4"]
tgt_dir =  os.path.abspath("/mnt/aoni02/nogawa/gaze/data/omni_dialog/real/speech")

for movie_file in movie_files:
    movie_id = movie_file.split(".")[0]
    movie_path = os.path.join(src_dir, movie_file)
    
    tgt_path = os.path.join(tgt_dir, "%s.mp3" % movie_id)
    
    command = "ffmpeg -y -i {0} -ab 128k {1}".format(movie_path, tgt_path).split(" ")
    print(command)
    
    print(' '.join(command))
    subprocess.call(command)
utils.notify("extract_speech done!!!")
