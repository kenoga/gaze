# -*- coding: utf-8 -*-

import os
import sys
import math
import glob
import json
import cv2
import numpy as np
import dlib
from openface import AlignDlib

from IPython.display import display, Image

DS_TYPES = {'movie', 'image', 'transformed_image', 'both_eyes', 'face_image', 'aligned_face'}
DS_ROOT_DIR = '/root/gaze/data/kobas-omni-eyecontact'


def imshow(img, format='.png'):
    decoded_bytes = cv2.imencode(format, img)[1].tobytes()
    display(Image(data=decoded_bytes))   


# src_dirの子ディレクトリ(sub_dir)の中に含まれる全ての画像(img)に対してfuncを適用してdst_dir/sub_dir/new_imgとして保存する
def apply_func_against_srcdir_and_save_to_dstdir(src_dir, dst_dir, func, give_fname=False, **kwargs):
    assert os.path.exists(src_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    
    paths = glob.glob(os.path.join(src_dir, '*', '*.jpg'))
    print(len(paths))
    img_num = len(paths)
    count = 1
    for path in paths:
        sys.stdout.write('\r'+"%d / %d" % (count, img_num))
        count += 1
        
        img = cv2.imread(path)
        # funcは新しいimgまたはNoneを返す
        if give_fname:
            new_img = func(img, os.path.basename(path), **kwargs)
        else:
            new_img = func(img, **kwargs)
        if new_img is None:
            continue
        # サブディレクトリ名の取得
        sub_dir = os.path.basename(os.path.dirname(path))
        fname = os.path.basename(path)
        dst_sub_dir = os.path.join(dst_dir, sub_dir)
        if not os.path.exists(dst_sub_dir):
            os.mkdir(dst_sub_dir)
        cv2.imwrite(os.path.join(dst_sub_dir, fname), new_img)
        

    
# person_id -> list of filenameのdictを返す
def get_image_paths(img_type):
    paths_dict = {}
    dsdir = os.path.join(DS_ROOT_DIR, img_type)
    subs = get_subdirs(img_type)
    
    for sub in subs:
        fpaths = [path for path in glob.glob(os.path.join(dsdir, sub, '*.jpg'))]
        paths_dict[sub] = fpaths        
    return paths_dict


# 指定されたディレクトリの子ディレクトリを返す
def get_subdirs(img_type='face_image'):
    return sorted([subdir for subdir in os.listdir(os.path.join(DS_ROOT_DIR, img_type)) if subdir[0] != '.'])



# person_id -> bb_and_landmarks dictのdictを返す
def get_bb_and_landmarks_dict():
    result = {}
    dir_path = os.path.join(DS_ROOT_DIR, 'face_bb_and_landmarks')
    jsons = glob.glob(os.path.join(dir_path, '*.json'))
    for json_file in jsons:
        json_path = os.path.join(DS_ROOT_DIR, json_file)
        with open(json_path, 'r') as fr:
            person_id = os.path.basename(json_file).split('.')[0]
            result[person_id] = json.load(fr)
    return result
            
class FaceAligner():
    def __init__(self):
        DLIB_MODEL_PATH = '/root/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
        self.aligner = AlignDlib(DLIB_MODEL_PATH)
        
    def align(self, img_path=None, img=None, bb=None, landmarks=None, size=255):
        if img_path is None and img is None:
            raise RuntimeError("Please give a imgpath or a img.")
        if img_path:
            img = cv2.imread(img_path)
        return self.aligner.align(255, img, bb=bb, landmarks=landmarks)
    
    def bb(self, img_path=None, img=None):
        if img_path is None and img is None:
            raise RuntimeError("Please give a imgpath or a img.")
        if img_path:
            img = cv2.imread(imgpath)
        return self.aligner.getLargestFaceBoundingBox(img)
    
    def landmarks(self, img_path=None, img=None, bb=None):
        if bb is None:
            raise RuntimeError("Please give a bb.")
        if img_path is None and img is None:
            raise RuntimeError("Please give a imgpath or a img.")
        if img_path:
            img = cv2.imread(imgpath)
        return self.aligner.findLandmarks(img, bb)


def get_circle_partition(img, radius):
    H, W, _ = img.shape
    rr, cc = draw.circle(H//2, W//2, radius)
    eimg = np.ones(img.shape, dtype=np.uint8) * 255
    eimg[rr, cc, :] = img[rr, cc, :]
    return eimg
 
    
class PolarTransformer():

    def __init__(self, h_src, w_src):
        self.h_src = h_src
        self.w_src = w_src
        
        self.x_center, self.y_center = h_src // 2, w_src//2
        
        self.w_dst = w_src * 2
        self.h_dst = h_src // 2
        
        # dst中のpxからsrc中のpxへのmappingをあらかじめ計算しておく
        self.__create_polar_map()
    
    def __create_polar_map(self):
        self.polar_map = [[0 for i in range(self.w_dst)] for j in range(self.h_dst)]
        
        for h_px in range(self.h_dst):
            r = h_px
            for w_px in range(self.w_dst):
                t = 2 * math.pi * (float(w_px + 1) / self.w_dst)
                x = r * math.cos(t)
                y = -r * math.sin(t)
                self.polar_map[h_px][w_px] = (y, x)
      
    def transform(self, src, h_top=0, h_bottom=1, w_left=0, w_right=1, interpolation=True):
        '''
        height * h_topからheight * h_bottomまでが変換対象になる
        width * w_leftからwidth * w_leftまでが変換対象になる
        '''
        assert h_top < h_bottom <= 1
        assert w_left < w_right <= 1
        dst = np.zeros((self.h_dst, self.w_dst, 3), np.uint8)
        
        top = int(self.h_dst * h_top)
        bottom = int(self.h_dst * h_bottom)
        left = int(self.w_dst * w_left)
        right = int(self.w_dst * w_right)
        for h_px in range(top, bottom):
            for w_px in range(left, right):
                src_y, src_x = self.polar_map[h_px][w_px]
                src_iy, src_ix = int(src_y), int(src_x)
                y = self.y_center + src_iy
                x = self.x_center + src_ix
                if not interpolation:
                    dst[h_px][w_px] = src[y][x]
                else:
                    if y > src.shape[0] - 2: y = src.shape[0] - 2
                    if x > src.shape[1] - 2: x = src.shape[1] - 2
                    dx = abs(src_x - src_ix)
                    dy = abs(src_y - src_iy)
                    dst[h_px][w_px] = (1-dx) * (1-dy) * src[y][x] \
                        + dx * (1-dy) * src[y][x+1] \
                        + (1-dx) * dy * src[y+1][x] \
                        + dx * dy * src[y+1][x+1]
        return dst[top:bottom, left:right]
    
    
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
    
    
## methods for pre-processing
def movies_to_images():
    src_dir = os.path.join(DS_ROOT_DIR, "movie")
    dst_dir = os.path.join(DS_ROOT_DIR, "image")
    subdirs = os.path.join(DS_ROOT_DIR)
    movie_paths = glob.glob(os.path.join(src_dir, '*', '*.MP4'))
    for movie_path in movie_paths:
        subdir_name = os.path.basename(os.path.dirname(movie_path))
        imgs_dir = os.path.join(dst_dir, subdir_name)
        mov2imgs(movie_path, imgs_dir)

def transform_images():
    def polar_transform_with_interpolation(img, fname, transformer=None):
        place = fname.split("_")[2]
        if place in {"A", "C"}:
            polar_img = polar_transformer.transform(img, 0.5, 1, 0.75, 1, interpolation=True)
        else:
            polar_img = polar_transformer.transform(img, 0.5, 1, 0.5, 0.75, interpolation=True)
        return polar_img

    def polar_transform(img, fname, transformer=None):
        place = fname.split("_")[2]
        if place in {"A", "C"}:
            polar_img = polar_transformer.transform(img, 0.5, 1, 0.75, 1, interpolation=False)
        else:
            polar_img = polar_transformer.transform(img, 0.5, 1, 0.5, 0.75, interpolation=False)
        return polar_img
    
    src_dir = os.path.join(DS_ROOT_DIR, 'image')
    dst_dir = os.path.join(DS_ROOT_DIR, 'transformed')

    print('initialization start')
    polar_transformer = PolarTransformer(2880, 2880)
    print('initialization end')

    apply_func_against_srcdir_and_save_to_dstdir(src_dir, \
                                                               dst_dir, \
                                                               polar_transform, \
                                                               give_fname=True, \
                                                               transformer=polar_transformer)
    

def detect_faces():
    # subdirごとに結果のjsonファイルを保存する
    json_fname = "%s.json"
    img_path_dict = get_image_paths('transformed')
    face_root_dir = os.path.join(DS_ROOT_DIR, "face")
    json_dir = os.path.join(DS_ROOT_DIR, "face_bb_and_landmarks")
    aligner = FaceAligner()
    
    data_size = sum([len(img_path_dict[subdir]) for subdir in img_path_dict.keys()])
    print("data_size: %d" % data_size)
    count = 0

    if not os.path.exists(json_dir):
        os.mkdir(json_dir)

    results = {}    
    for sub_dir in sorted(img_path_dict.keys()):
        print(sub_dir)
        img_paths = img_path_dict[sub_dir]
        face_dir = os.path.join(face_root_dir, sub_dir)    

        if not os.path.exists(face_dir):
            os.makedirs(face_dir)

        for img_path in img_paths:
            count += 1
            sys.stdout.write('\r'+"person: %d / %d, all: %d / %d" % (count, len(img_paths), count, data_size))
            img_name = os.path.basename(img_path)
            results[img_name] = {}
            img = cv2.imread(img_path)
            bb = aligner.bb(img=img)

            if bb is not None:
                landmarks = aligner.landmarks(img=img, bb=bb)
                # save face img
                face_img = img[bb.top():bb.bottom(), bb.left():bb.right()]
                cv2.imwrite(os.path.join(face_dir, img_name), face_img)
                results[img_name]["detected"] = True
                results[img_name]['bb'] = [bb.left(), bb.top(), bb.right(), bb.bottom()]
                results[img_name]["landmarks"] = landmarks
            else:
                results[img_name]["detected"] = False

        with open(os.path.join(json_dir, json_fname) % sub_dir, "w") as fr:
            json.dump(results, fr, indent=2, sort_keys=True)
    
    
def align_faces():
    aligner = FaceAligner()
    pid2bblms = get_bb_and_landmarks_dict()
    src_dir = os.path.join(DS_ROOT_DIR, 'transformed')
    dst_dir = os.path.join(DS_ROOT_DIR, 'aligned_face')

    def align_face(img, img_name, pid2bblms=None):
        if pid2bblms is None:
            return None
        pid = img_name.split("_")[0]
        if img_name in pid2bblms[pid] and pid2bblms[pid][img_name]['detected']:
            bb = pid2bblms[pid][img_name]['bb']
            bb_rec = dlib.rectangle(bb[0], bb[1], bb[2], bb[3])
            lms = pid2bblms[pid][img_name]['landmarks']
            return  aligner.align(img=img, bb=bb_rec, landmarks=lms)
        return None
    
    apply_func_against_srcdir_and_save_to_dstdir(src_dir, dst_dir, align_face, give_fname=True, pid2bblms=pid2bblms)
    
    
def extract_eyes():
    def extract_both_eyes_from_aligned_face_img(img):
        height = img.shape[0]
        return img[int(height * 0.075):int(height * 0.25)]    
    src_dir = os.path.join(DS_ROOT_DIR, "aligned_face")
    dst_dir = os.path.join(DS_ROOT_DIR, "both_eyes_from_aligned_face")

    apply_func_against_srcdir_and_save_to_dstdir(src_dir, dst_dir, extract_both_eyes_from_aligned_face_img)
    