from gaze_detector import GazeDetector
from train.model.cnn import CNN

model_path = "./train/trained/training_01/training_01_00.npz"
model = CNN()

img_path = "./data/omni_dialog/pilot/both_eyes_from_aligned_face/omni_dialog_pilot_01/00000001_A.jpg"

gaze_detector = GazeDetector(model, model_path)
result = gaze_detector.detect(img_path)
print(result)
