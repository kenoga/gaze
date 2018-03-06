
import os
import numpy as np
from io import BytesIO
from skimage import io, draw
import codecs
import cv2

def extract_under_half(frame):
    height = frame.shape[0]
    return frame[0:height//2, :]

def mov2imgs(in_path, out_path, func):
    # func: 動画から得られた画像に対してかける処理
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise Exception("動画の読み込みに失敗しました。")
        
    in_movie_name = os.path.basename(in_path)
    frame_count = 0
    
    while True:
        # 5フレームおきに保存する
        for _ in range(0,4):
            _, _ = cap.read()
        ret, frame = cap.read()
        frame_count += 5
        if not ret:
            break
        
        frame = func(frame)
        img_name = ".".join([in_movie_name, '%08d' % frame_count, "jpg"])
        cv2.imwrite(os.path.join(out_path, img_name), frame)
        print(frame_count)
    cap.release()
    
in_path = './movie/test_para.mp4'
out_dir = './para_image'
mov2imgs(in_path, out_dir, extract_under_half)
