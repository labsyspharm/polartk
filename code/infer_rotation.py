import tifffile

import matplotlib.pyplot as plt

from pyprojroot import here
from skimage import registration, transform


# paths
fixed_filepath = here("./data/single_cell.tif")
moving_filepath = here("./data/case_rotation.tif")

# load data
fixed_img = tifffile.imread(fixed_filepath)
moving_img = tifffile.imread(moving_filepath)

# infer
fixed_img = fixed_img[0, ...]
moving_img = moving_img[0, ...]

# convert to polar coordinate
radius = fixed_img.shape[0] // 2
fixed_img = transform.warp_polar(fixed_img, radius=radius)
moving_img = transform.warp_polar(moving_img, radius=radius)

shifts = registration.phase_cross_correlation(
        reference_image=fixed_img,
        moving_image=moving_img,
        upsample_factor=1,
        return_error=False,
        )
shifts = {axis: val for axis, val in zip(["degree", "radius"], shifts)}
print(f"inferred transform: {shifts}")
