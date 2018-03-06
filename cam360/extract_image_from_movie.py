import cv2
import numpy as np
import pdb
import math
import os
import datetime


def extract_image_from_movie_man(in_path, out_path):
    movie_name = os.path.basename(in_path)
    cap = cv2.VideoCapture(in_path)

    frame_count = 0
    while True:
        frame_count += 1
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            continue
        elif key == ord('s'):
            img_name = '.'.join([movie_name, '%04d' % frame_count, 'jpg'])
            print(img_name)
            cv2.imwrite(os.path.join(out_path, img_name), frame)

    cap.release()
    cv2.destroyAllWindows()

extract_image_from_movie_man('./movie/test1.mp4', './raw_image')
