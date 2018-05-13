# -*- coding: utf-8 -*-
import os
import numpy as np
from io import BytesIO
import codecs
import cv2

def extract_under_half(frame):
    height = frame.shape[0]
    return frame[0:height//2, :]
    
def extract_left(frame):
    width = frame.shape[1]
    return frame[:, 0:width//2]

def mov2imgs(in_path, out_path, func=lambda x: return x, interval=2):
#     print('test')
    # func: 動画から得られた画像に対してかける処理
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise Exception("動画の読み込みに失敗しました。")
        
    in_movie_name = os.path.basename(in_path)
    frame_count = 0
    
    while True:
        # interval+1フレームおきに保存する
        for _ in range(0,interval):
            _, _ = cap.read()
        ret, frame = cap.read()
        frame_count += 5
        if not ret:
            break
        
        frame = func(frame)
#         img_name = "_".join([in_movie_name.split('.')[0],  "%08d" % fname_count]) + ".jpg"
        print(in_movie_name + "->" + img_name)
        
#         cv2.imwrite(os.path.join(out_path, img_name), frame)
#         print(frame_count)
    cap.release()
    
# in_path = '../data/movie/test_omni.mp4'
# out_dir = '../data/omni_image'
# mov2imgs(in_path, out_dir, extract_left)
