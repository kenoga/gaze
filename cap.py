
import cv2
import numpy as np
import math
import os
import datetime

# in_image_height = 2000
# in_image_width = 2000
# fps = 50

path = "./testimage/%s.jpg"
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, in_image_height)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, in_image_width)
# cap.set(cv2.CAP_PROP_FPS, fps)

while True:
    time = "{0:%Y%m%d%H%M%S%f}".format(datetime.datetime.now())
    ret, frame = cap.read()
    cv2.imwrite(path % time, frame)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(20)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('exit')