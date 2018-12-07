import os, sys
import glob
sys.path.append("..")
sys.path.append("../..")
from utils.PolarTransformer import PolarTransformer
import dataset_utils.utils as utils
import dataset_utils.config as config
import omni_2_panorama

polar_transformer = PolarTransformer(2880, 2880)

src_dir = os.path.join(config.DS_DIALOG_PILOT, "image")
tgt = os.path.join(config.DS_DIALOG_PILOT, "transformed")

if not os.path.exists(tgt):
    os.mkdir(tgt)
    
src_paths = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))
for i, path in enumerate(src_paths):
    print("%d / %d" % (i+1, len(src_paths)))
        
    fname = os.path.basename(path)
    fid, fmt = tuple(fname.split("."))
    
    img = utils.imread(path)
    
    for place in ["A", "B"]:
        t_img = omni_2_panorama.polar_transform_by_place(img, place, polar_transformer)
        t_img_id = "_".join([fid, place])
        t_img_name = ".".join([t_img_id, fmt])
        t_img_path = os.path.join(tgt, t_img_name)
        utils.imwrite(t_img_path, t_img)
        print("%s -> %s" % (path, t_img_path))
