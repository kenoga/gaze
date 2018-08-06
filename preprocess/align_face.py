# -*- coding: utf-8 -*-

import os
import sys
import dlib
import openface

module_path = os.path.abspath("..")
if module_path not in sys.path:
    sys.path.append(module_path)

from utils.PolarTransformer import PolarTransformer
from utils.FaceAligner import FaceAligner
from utils.utils import apply_func_against_srcdir_and_save_to_dstdir
from dataset_utils.utils import get_bb_and_landmarks_dict
from dataset_utils.config import DS_ROOT


def main():
    aligner = FaceAligner()
    pid2bblms = get_bb_and_landmarks_dict()
    src_dir = os.path.join(DS_ROOT, 'transformed')
    dst_dir = os.path.join(DS_ROOT, 'aligned_face2')

    def align_face(img, img_name, pid2bblms=None, size=None, indices=None):
        assert pid2bblms is not None
        assert size is not None
        assert indices is not None

        pid = img_name.split("_")[0]
        if img_name in pid2bblms[pid] and pid2bblms[pid][img_name]['detected']:
            bb = pid2bblms[pid][img_name]['bb']
            bb_rec = dlib.rectangle(bb[0], bb[1], bb[2], bb[3])
            lms = pid2bblms[pid][img_name]['landmarks']
            return  aligner.align(img=img, bb=bb_rec, landmarks=lms, size=size, indices=indices)
        return None
    
    apply_func_against_srcdir_and_save_to_dstdir(src_dir, dst_dir, align_face, \
                                                 give_fname=True, \
                                                 pid2bblms=pid2bblms, \
                                                 size=96, \
                                                 indices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    
    
main()


