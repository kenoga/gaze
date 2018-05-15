import cv2
import os
import argparse
import json


def detect_faces_from_dir(src_dir, face_detector, log_fw):
    results = {}
    results['dir_absolute_path'] = os.path.abspath(args.src_dir)
    results['results'] = {}
    
    src_fnames = [fname for fname in os.listdir(args.src_dir) if fname[0] != '.']
    for src_fname in src_fnames:
        src_path = os.path.abspath(os.path.join(args.src_dir, src_fname))
        print(src_path)
        
        img = cv2.imread(src_path)
        faces = fd.detect(img)
        print(faces)
        print(type(faces))
        results['results'][src_fname] = faces
        
        log_fw.write(src_path + ' ' + str(faces) + '\n')
    return results


if __name__ == "__main__":
    # コマンドライン引数で以下を指定する
    # - 顔検出をしたい画像が入っているディレクトリ (src_dir)
    # - 結果のjsonファイルの書き出し先ディレクトリ (dest_dir)
    parser = argparse.ArgumentParser()
    parser.add_argument('src_dir', help='a source directory that contains input images')
    parser.add_argument('dest_dir', help='a directory where you put log files')
    parser.add_argument('-m', '--model', choices=['opencv', 'dlib_svm', 'dlib_cnn'], default='opencv')
    args = parser.parse_args()

    # 途中でプログラムが終了したときのためにlogを書き出しておくためのfw
    log_file_name = os.path.basename(args.src_dir) + '.facedetection.%s.log' % args.model
    log_file_path = os.path.join(args.dest_dir, log_file_name)
    log_fw = open(log_file_path, 'w')

    # 結果をjson形式で書き出し
    if args.model == 'opencv':
        from face_detectors.face_detector_opencv_cascade import FaceDetectorOpenCVCascade
        fd = FaceDetectorOpenCVCascade()
    elif args.model == 'dlib_svm':
        from face_detectors.face_detector_dlib_svm import FaceDetectorDlibSVM
        fd = FaceDetectorDlibSVM()
    else:
        print('%s has not implemented yet...' % args.model)
        import sys
        sys.exit(0)
    results = detect_faces_from_dir(args.src_dir, fd, log_fw)

    result_file_name = os.path.basename(args.src_dir) + '.facedetection.%s.json' % args.model
    result_file_path = os.path.join(args.dest_dir, result_file_name)
    with open(result_file_path, 'w') as fw:
        json.dump(results, fw, indent=2, sort_keys=True)
    log_fw.close()

