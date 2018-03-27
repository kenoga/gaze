
import cv2
import numpy as np
import pdb
import math
import os


# default: 2160, 3840, 3
# in_image_height = 2160
# in_image_width = 3840
in_image_height = 2000
in_image_width = 2000
fps = 50

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, in_image_height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, in_image_width)
cap.set(cv2.CAP_PROP_FPS, fps)

out = cv2.VideoWriter('output1.m4v', 
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps,
                      (in_image_height,in_image_width))

ret, frame = cap.read()
print('height: %d, width: %d, channel: %d' % (frame.shape[0], frame.shape[1], frame.shape[2]))

while True:
    if not cap.isOpened():
        raise Exception('カメラを初期化できませんでした.カメラのモード,または電源を確認してください.')
    
    ret, frame = cap.read()
    out.write(frame)
    # cv2.imshow('test', frame)
    
    key = cv2.waitKey(20)
    if key & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
print('exit')