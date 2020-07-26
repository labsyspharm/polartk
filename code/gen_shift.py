import numpy as np
import tifffile

from skimage import transform
from pyprojroot import here


# paths
src_filepath = here("./data/single_cell.tif")

# transform cases
translation = transform.AffineTransform(translation=(2, -3))
rotation = transform.AffineTransform(rotation=np.radians(35))
case_dict = {
    "translation": [translation],
    "rotation": [rotation],
    "translation_rotation": [translation, rotation],
}

# go through cases
src_img = tifffile.imread(src_filepath)
for case in case_dict:
    # warp image
    dst_img = src_img.copy()
    for ch in range(dst_img.shape[0]):
        tmp_img = dst_img[ch, ...].copy()
        for trans in case_dict[case]:
            tmp_img = transform.warp(
                image=tmp_img,
                inverse_map=trans.inverse,
                mode="constant",
                cval=0.0,
                clip=True,
                preserve_range=True,
            ).astype(dst_img.dtype)
        dst_img[ch, ...] = tmp_img
    # save to disk
    dst_filepath = here(f"./data/case_{case}.tif")
    tifffile.imsave(dst_filepath, dst_img)
