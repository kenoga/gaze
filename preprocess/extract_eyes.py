
import os

module_path = os.path.abspath("..")
if module_path not in sys.path:
    sys.path.append(module_path)
    
from utils.utils import apply_func_against_srcdir_and_save_to_dstdir
from dataset_utils.config import DS_ROOT


def extract_eyes():
    def extract_both_eyes_from_aligned_face_img(img):
        height = img.shape[0]
        return img[int(height * 0.075):int(height * 0.25)]    
    src_dir = os.path.join(DS_ROOT, "aligned_face")
    dst_dir = os.path.join(DS_ROOT, "both_eyes_from_aligned_face")

    apply_func_against_srcdir_and_save_to_dstdir(src_dir, dst_dir, extract_both_eyes_from_aligned_face_img)
    