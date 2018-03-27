
import cv2
import numpy as np
import math
import os

def polar_transform(src):
    # src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    h_src, w_src, _ = src.shape

    x_center, y_center = h_src // 2, w_src//2
    y_center = w_src // 2
    
    # theta
    w_dst = w_src * 2
    # radius
    h_dst = h_src // 2
    
    dst = np.zeros((h_dst, w_dst, 3), np.uint8)

    for h_px in range(h_dst):
        r = h_px
        for w_px in range(w_dst):
            t = 2 * math.pi * ((w_px + 1) / w_dst)
            x = int(r * math.cos(t))
            y = -int(r * math.sin(t))
            dst[h_px][w_px] = src[y_center + y][x_center + x]
    return dst


# # radius
# w_dst = w_src // 2
# # theta
# h_dst = h_src
# 
# dst = np.zeros((h_dst, w_dst), np.uint8)
# 
# for h_px in range(h_dst):
#     t = 2 * math.pi * ((h_px + 1) / h_dst)
#     for w_px in range(w_dst):
#         r = w_px
#         x = int(r * math.cos(t))
#         y = int(r * math.sin(t))
#         dst[h_px][w_px] = src[y_center + y][x_center + x]
# 
# cv2.imwrite('../data/polar.jpg', dst)
# 




# src = cv2.imread('../data/test/testout.jpg')
# dst = polar_transform(src)
# cv2.imwrite('../data/polar.jpg', dst)

src_dir = '../data/omni_image'
src_files = os.listdir(src_dir)
src_paths = [os.path.join(src_dir, src_file) for src_file in src_files if src_file[0] != '.']

out_dir = '../data/para_image_polar'
for path in src_paths:
    print(path)
    src = cv2.imread(path)
    dst = polar_transform(src)
    cv2.imwrite(os.path.basename(path), dst)


