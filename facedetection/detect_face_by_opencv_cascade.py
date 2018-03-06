import cv2
import numpy as np
import math
import os
import datetime

class FaceDetector():
    def __init__(self):
        model_file_path = "/Users/nogaken/.pyenv/versions/anaconda3-4.0.0/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
        self.detector = cv2.CascadeClassifier(model_file_path)

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.3, 5)
        return faces

image_dir = '../cam360/para_image'
image_paths = sorted([os.path.join(image_dir, file_name) for file_name in os.listdir(image_dir) if file_name[0] != '.'])

fw = open('./results/face_detection_result_para_image_opencv.log', 'w')
fd = FaceDetector()
for image_path in image_paths:
    img = cv2.imread(image_path)
    faces = fd.detect(img)
    result = ''
    if len(faces) == 0:
        result = 'None'
    else:
        result = ','.join(['[' + ','.join(map(str, face.tolist())) + ']' for face in faces])
    print(image_path, result)
    fw.write(' '.join([image_path, result]) + '\n')
