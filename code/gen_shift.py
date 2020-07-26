import numpy as np
import tifffile

from skimage import transform
from pyprojroot import here


# paths
src_filepath = here("./data/single_cell.tif")

# transform cases
def move_translation(input):
    tform = transform.AffineTransform(translation=(2, -3))
    return transform.warp(input, tform, mode="constant", cval=0.0, clip=True,
            preserve_range=True).astype(input.dtype)
def move_rotation(input):
    return transform.rotate(input, 35, mode="constant", cval=0.0,
            clip=True, preserve_range=True).astype(input.dtype)

case_dict = {
    "translation": [move_translation],
    "rotation": [move_rotation],
    "translation_rotation": [move_translation, move_rotation],
}

# go through cases
src_img = tifffile.imread(src_filepath)
for case in case_dict:
    # warp image
    dst_img = src_img.copy()
    for ch in range(dst_img.shape[0]):
        tmp_img = dst_img[ch, ...].copy()
        for trans_fn in case_dict[case]:
            tmp_img = trans_fn(tmp_img)
        dst_img[ch, ...] = tmp_img
    # save to disk
    dst_filepath = here(f"./data/case_{case}.tif")
    tifffile.imsave(dst_filepath, dst_img)
