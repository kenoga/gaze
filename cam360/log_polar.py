
import cv2
import numpy as np

src = cv2.imread('../data/test/testout.jpg')
h, w, _ = src.shape
# r = h // 2

out_img = cv2.logPolar(src, (h/2, w/2), 350, cv2.WARP_FILL_OUTLIERS)
out_img = cv2.logPolar(src, (h/2, w/2), 350, cv2.WARP_FILL_OUTLIERS)
cv2.logPolar
print(out_img.shape)
cv2.imshow('log polar', out_img)
cv2.imwrite('./lp_test.jpg', out_img)
cv2.waitKey(0)