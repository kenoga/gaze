# -*- coding: utf-8 -*-

import os

module_path = os.path.abspath("..")
if module_path not in sys.path:
    sys.path.append(module_path)
    
from utils.PolarTransformer import PolarTransformer
from dataset_utils.config import DS_ROOT


def polar_transform_with_interpolation(img, fname, transformer=None):
    place = fname.split("_")[2]
    if place in {"A", "C"}:
        polar_img = polar_transformer.transform(img, 0.5, 1, 0.75, 1, interpolation=True)
    else:
        polar_img = polar_transformer.transform(img, 0.5, 1, 0.5, 0.75, interpolation=True)
    return polar_img

def polar_transform(img, fname, transformer=None):
    place = fname.split("_")[2]
    if place in {"A", "C"}:
        polar_img = polar_transformer.transform(img, 0.5, 1, 0.75, 1, interpolation=False)
    else:
        polar_img = polar_transformer.transform(img, 0.5, 1, 0.5, 0.75, interpolation=False)
    return polar_img
    
def main():
    src_dir = os.path.join(DS_ROOT, 'image')
    dst_dir = os.path.join(DS_ROOT, 'transformed')

    print('initialization start')
    polar_transformer = PolarTransformer(2880, 2880)
    print('initialization end')

    apply_func_against_srcdir_and_save_to_dstdir(src_dir, \
                                                dst_dir, \
                                                polar_transform, \
                                                give_fname=True, \
                                                transformer=polar_transformer)
    
main()