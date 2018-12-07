import os, sys, glob, json
from gaze_detector import GazeDetector
from train.model.cnn import CNN

model_path = "./train/trained/training_01/training_01_00.npz"
model = CNN()
gaze_detector = GazeDetector(model, model_path)


src_dir = "./data/omni_dialog/pilot/both_eyes_from_aligned_face"
# movie_ids = ["omni_dialog_pilot_01"]
movie_ids = ["omni_dialog_pilot_02_01", "omni_dialog_pilot_02_02"]
tgt_dir = "./data/omni_dialog/pilot/detection_result"

if not os.path.exists(tgt_dir):
    os.makedirs(tgt_dir)

for movie_id in movie_ids:
    results = {}
    json_path = os.path.join(tgt_dir, "%s.json" % movie_id)
    src_sub_dir = os.path.join(src_dir, movie_id)
    img_paths = sorted(glob.glob(os.path.join(src_sub_dir, "*.jpg")))
    for img_path in img_paths:
        result = gaze_detector.detect(img_path)
        img_name = os.path.basename(img_path)
        results[img_name] = result
        print("%s -> %s" % (img_name, str(result)))
    with open(json_path, "w") as fw:
        json.dump(results, fw)
