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
from dataset_utils.config import DS_ROOT, DS_KATAYAMA


def main(src_dir, dst_dir, dataset_path=DS_ROOT):
    aligner = FaceAligner()
    bblms = get_bb_and_landmarks_dict(dataset_path=dataset_path)

    def align_face(img, img_name, bblms=None, size=None, indices=None):
        assert bblms is not None
        assert size is not None
        assert indices is not None

        if img_name in bblms:
            bb = bblms[img_name]['bb']
            bb_rec = dlib.rectangle(bb[0], bb[1], bb[2], bb[3])
            lms = bblms[img_name]['landmarks']
            return  aligner.align(img=img, bb=bb_rec, landmarks=lms, size=size, indices=indices)
        return None
    
    apply_func_against_srcdir_and_save_to_dstdir(src_dir, dst_dir, align_face, \
                                                 format='png', \
                                                 give_fname=True, \
                                                 bblms=bblms, \
                                                 size=96, \
                                                 indices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

src_dir = os.path.join(DS_KATAYAMA, 'image')
dst_dir = os.path.join(DS_KATAYAMA, 'aligned_face')
dataset_path = DS_KATAYAMA
main(src_dir, dst_dir, dataset_path)


