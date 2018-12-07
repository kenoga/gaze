# -*- coding: utf-8 -*-

import os
import sys
import glob
import cv2

# src_dirの子ディレクトリ(sub_dir)の中に含まれる全ての画像(img)に対してfuncを適用してdst_dir/sub_dir/new_imgとして保存する
def apply_func_against_srcdir_and_save_to_dstdir(src_dir, dst_dir, func, format='jpg', give_fname=False, **kwargs):
    assert os.path.exists(src_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    
    paths = glob.glob(os.path.join(src_dir, '*', '*.%s' % format))
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
