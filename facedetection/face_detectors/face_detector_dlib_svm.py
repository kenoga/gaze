import dlib

class FaceDetectorDlibSVM():
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, img):
        faces, scores, types = self.detector.run(img, 0, -0.1)
        return faces