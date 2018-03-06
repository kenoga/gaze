
import cv2
import numpy as np
import pdb
import math
import os

# default: 2160, 3840, 3
in_image_height = 2000
in_image_width = 2000

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, in_image_height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, in_image_width)
# cap.set(cv2.CAP_PROP_FPS, 30)

ret, frame = cap.read()
print("height: %d, width: %d, channel: %d" % (frame.shape[0], frame.shape[1], frame.shape[2]))
print(frame.shape)

while True:
    if not cap.isOpened():
        raise Exception("カメラを初期化できませんでした.カメラのモード,または電源を確認してください.")
    
    ret, frame = cap.read()
    cv2.imshow("test", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("end")