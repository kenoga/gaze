# -*- coding: utf-8 -*-

import os
import glob
import cv2

module_path = os.path.abspath("..")
if module_path not in sys.path:
    sys.path.append(module_path)
    
from dataset_utils.config import DS_ROOT


def mov2imgs(movie_path, imgs_dir, func=None, interval=2):
    # func: 動画から得られた画像に対してかける処理
    cap = cv2.VideoCapture(movie_path)
    if not cap.isOpened():
        raise Exception("動画の読み込みに失敗しました。")
    
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)
    movie_name = os.path.basename(movie_path)
    frame_count = 0
    
    while True:
        # interval+1フレームおきに保存する
        for _ in range(0,interval):
            _, _ = cap.read()
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            break
        
        if func:
            frame = func(frame)
        img_name = "_".join([movie_name.split('.')[0],  "%08d" % frame_count]) + ".jpg"
        print(movie_name + "->" + img_name)
        
        cv2.imwrite(os.path.join(imgs_dir, img_name), frame)
    cap.release()

def main():
    src_dir = os.path.join(DS_ROOT, "movie")
    dst_dir = os.path.join(DS_ROOT, "image")
    subdirs = os.path.join(DS_ROOT)
    movie_paths = glob.glob(os.path.join(src_dir, '*', '*.MP4'))
    for movie_path in movie_paths:
        subdir_name = os.path.basename(os.path.dirname(movie_path))
        imgs_dir = os.path.join(dst_dir, subdir_name)
        mov2imgs(movie_path, imgs_dir)
        
main()