import os
import sys
import argparse
import cv2
import dlib
import numpy as np
import scipy.spatial
import alignment

fileDir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, help='path to video')
parser.add_argument('--dimension', type=int, default=96, help='alignment dimension')
args = parser.parse_args()

videoCapture = cv2.VideoCapture(args.video)

if not videoCapture.isOpened():
    sys.exit('video not opened')

template = np.load(os.path.join(fileDir, 'template.npy'))
delaunay = scipy.spatial.Delaunay(template)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(fileDir, 'shape_predictor_68_face_landmarks.dat'))

alignment = alignment.Alignment(args.dimension, template, delaunay.simplices)

while True:

    ret, rawImage = videoCapture.read()

    if not ret:
        break

    boundingBox = max(detector(rawImage, 1), key=lambda rect: rect.width() * rect.height())
    landmarks = list(map(lambda point: (point.x, point.y),
                         predictor(rawImage, boundingBox).parts()))

    alignedImage = alignment.align(rawImage, landmarks)

    cv2.imshow('aligned image', alignedImage)

    if cv2.waitKey(1) == ord('q'):
        break

videoCapture.release()
