
import os
import numpy as np
from io import BytesIO
from skimage import io, draw
import codecs

def get_circle_partition(img, radius):
    H, W, _ = img.shape
    rr, cc = draw.circle(H//2, W//2, radius)
    eimg = np.ones(img.shape, dtype=np.uint8) * 255
    eimg[rr, cc, :] = img[rr, cc, :]
    return eimg
    
in_path = './raw_image'
out_dir = './circle_image'

image_names = os.listdir(in_path)

for in_image_name in os.listdir(in_path):
    in_image_path = os.path.join(in_path, in_image_name)
    img = io.imread(in_image_path)
    circle = get_circle_partition(img, 1400)
    io.imsave(os.path.join(out_dir, in_image_name), circle)
    