
opencv_path = './results/face_detection_result_para_image_opencv.log'
dlib_path = './results/face_detection_result_para_image_dlib.log'
paths = [opencv_path, dlib_path]

for path in paths:
    fr = open(path, 'r')
    lines = fr.readlines()

    failed_count = 0
    total_count = len(lines)
    for line in lines:
        face_positions = line.split(' ')[1].strip()
        if face_positions == 'None':
            failed_count += 1

    recall = (total_count - failed_count) / float(total_count)
    print(path)
    print('recall: %.4f' % recall)