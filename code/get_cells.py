from pathlib import Path

import tifffile
import numpy as np
import pandas as pd

from skimage import measure


# paths
data_folderpath = Path("/scratch/work/data_polartk")
img_filepath = data_folderpath / "image.ome.tif"
cellmask_filepath = data_folderpath / "cell_mask.tif"
marker_filepath = data_folderpath / "markers.csv"

# config
markers = ["DNA1", "PD1"]
patch_size = (15, 15)

# calculate coordinates
cellmask = tifffile.imread(cellmask_filepath)
regions = pd.DataFrame(
    measure.regionprops_table(label_image=cellmask, properties=["label", "centroid"])
).values
bounds = np.empty((regions.shape[0], 4), dtype=int)
bounds[:, 0] = np.rint(regions[:, 1] - patch_size[0] // 2).astype(int)
bounds[:, 2] = np.rint(regions[:, 2] - patch_size[1] // 2).astype(int)
bounds[:, 1] = bounds[:, 0] + patch_size[0]
bounds[:, 3] = bounds[:, 2] + patch_size[1]

# remove boundary cases
mask = (bounds[:, 0] < patch_size[0])\
        | (bounds[:, 2] < patch_size[1])\
        | (bounds[:, 1] + patch_size[0] > cellmask.shape[0])\
        | (bounds[:, 3] + patch_size[1] > cellmask.shape[1])
keep = np.logical_not(mask)
regions, bounds = regions[keep, :], bounds[keep, :]

# vectorized cropping
def crop(arr: np.ndarray):
    fn = lambda x: arr[x[0] : x[1], x[2] : x[3]]
    return np.apply_along_axis(func1d=fn, axis=1, arr=bounds)

# loop over markers
with marker_filepath.open() as f:
    marker_list = [line.strip() for line in f]

for name in markers:
    # crop to stack
    index = marker_list.index(name)
    with tifffile.TiffFile(img_filepath) as tif:
        img = tif.series[0].pages[index].asarray()
    patch_stack = crop(img)
    # save to disk
    out_filepath = data_folderpath / f"{name}.npy"
    np.save(out_filepath, patch_stack)
