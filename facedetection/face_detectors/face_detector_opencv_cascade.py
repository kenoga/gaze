import cv2

class FaceDetectorOpenCVCascade():
    def __init__(self):
        model_file_path = "/Users/nogaken/.pyenv/versions/anaconda3-4.0.0/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
        self.detector = cv2.CascadeClassifier(model_file_path)

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return []
        else:
            return faces.tolist()
