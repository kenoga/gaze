
import cv2
import os

def extract_face(img, x, y, w, h):
    return img[y:y+h, x:x+w]

def read_fd_results(path):
    results = []
    with open(path, 'r') as fr:
        for line in fr.readlines():
            line_s = line.strip().split(' ')
            results.append((line_s[0], line_s[1]))
    return results
            
path = './results/face_detection_result_para_image_opencv.log'
results = read_fd_results(path)

out_dir = "../data/para_image_face"

for result in results:
    path = result[0]
    box = result[1]
    
    basename = os.path.basename(path)
    
    if box == 'None':
        continue
    
    img = cv2.imread(os.path.join('../data/para_image', basename))
    # 一つの顔のみ検出できる
    x, y, w, h = tuple([int(e) for e in box.replace('[','').replace(']','').split(',')] )
    face = img[y:y+h, x:x+w]
    
    cv2.imwrite(os.path.join(out_dir, basename+'.face.jpg'), face)

out_dir = "../data/para_image_face"



