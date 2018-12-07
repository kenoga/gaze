# -*- coding: utf-8 -*-
import os, sys
import subprocess

import glob
sys.path.append("..")
sys.path.append("../..")
import dataset_utils.utils as utils
import dataset_utils.config as config

frame_rate = 20
movie_files = ["01_01.MP4", "01_02.MP4", "02_01.MP4", "02_02.MP4"]
src_img_dir = os.path.abspath("/mnt/aoni02/nogawa/gaze/data/omni_dialog/real/transformed_split")
src_spc_dir =  os.path.abspath("/mnt/aoni02/nogawa/gaze/data/omni_dialog/real/speech")
dst_dir = os.path.abspath("/mnt/aoni02/nogawa/gaze/data/omni_dialog/real/ant_movie")


for movie_file in movie_files:
    movie_id = movie_file.split(".")[0]
    sub_src_img_dir = os.path.join(src_img_dir, movie_id)
    src_spc_path = os.path.join(src_spc_dir, "%s.mp3" % movie_id)

    for place in ["A", "B", "C", "D"]:
        src_img_path = os.path.join(sub_src_img_dir, "%08d_{0}.png".format(place))
        tmp_dst_path = os.path.join(dst_dir, "{0}_{1}_tmp.mp4".format(movie_id, place))
        dst_path = os.path.join(dst_dir, "{0}_{1}.mp4".format(movie_id, place))
        images_to_movie = "ffmpeg -r {0} -i {1} -pix_fmt yuv420p {2}".format(frame_rate, src_img_path, tmp_dst_path).split(" ")
        print(' '.join(images_to_movie))
        subprocess.call(images_to_movie)
        attach_speech = "ffmpeg -i {0} -i {1} -vcodec copy -acodec aac {2}".format(tmp_dst_path, src_spc_path, dst_path).split(" ")
        print(' '.join(attach_speech))
        subprocess.call(attach_speech)
        os.remove(tmp_dst_path)
        
utils.notify("make_movie done!!!")
