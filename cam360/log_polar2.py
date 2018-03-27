
import cv2
import numpy as np
import math

src = cv2.imread('../data/test/testout.jpg')
src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

h_src, w_src = src.shape

x_center = h_src // 2
y_center = w_src // 2

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

# theta
w_dst = w_src
# radius
h_dst = h_src // 2

dst = np.zeros((h_dst, w_dst), np.uint8)

M = 100
for h_px in range(h_dst):
    # rho = M * math.log(h_px+1)
    # r = math.pow(math.e, rho)
    # rho = math.pow(math.e, h_px / M)
    r = math.pow(math.e, h_px/M)
    for w_px in range(w_dst):
        t = 2 * math.pi * ((w_px + 1) / w_dst)
        x = int(r * math.cos(t))
        y = int(r * math.sin(t))
        print(x,y)
        if y_center + y >= h_dst or y_center + y < 0 or x_center + x >= w_dst or x_center + x < 0:
            continue
        dst[h_px][w_px] = src[y_center + y][x_center + x]

cv2.imwrite('../data/log_polar.jpg', dst)
