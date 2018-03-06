
import numpy as np
import math
import os
import datetime
import dlib
from skimage import io, draw

class FaceDetector():
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, img):
        faces, scores, types = self.detector.run(img, 0, -0.1)
        return faces
        

test_path = '../cam360/para_image/test_para.mp4.00000100.jpg'
image_dir = '../cam360/para_image'
image_paths = sorted([os.path.join(image_dir, file_name) for file_name in os.listdir(image_dir) if file_name[0] != '.'])

# fw = open('face_detection_result_para_image.log', 'w')
fd = FaceDetector()

fw = open('./results/face_detection_result_para_image_dlib.log', 'w')
for image_path in image_paths:
    img = io.imread(image_path)
    recs = fd.detect(img)
    result = ''
    if len(recs) == 0:
        result = 'None'
    else:
        for rec in recs:
            result = ','.join(
                ['[' + ','.join(map(str, [rec.left(), rec.top(), rec.right()-rec.left(), rec.bottom()-rec.top()])) + ']' for rec in recs]
                )
    print(image_path, result)
    fw.write(' '.join([image_path, result]) + '\n')
