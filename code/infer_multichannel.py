# for each channel, infer translation and rotation independently
# then find the "consensus" using RANSAC algorithm in scikit-learn

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tifffile

from scipy import stats
from skimage import registration, transform


# paths
data_filepath = Path("/scratch/work/data_polartk/DNA1.npy")

# load data
data = np.load(data_filepath)

# make reference image
x, y = np.meshgrid(range(data.shape[1]), range(data.shape[2]))
grid = np.stack([x.flat, y.flat], axis=-1)
mu = (data.shape[1] / 2, data.shape[2] / 2)
cov = np.diag(data.shape[1:]) / 2
template = stats.multivariate_normal(mean=mu, cov=cov)
template_img = template.pdf(grid).reshape(x.shape)

# vectorized alignment
def infer_1(arr: np.ndarray):
    return registration.phase_cross_correlation(
            reference_image=template_img,
            moving_image=arr.reshape(data.shape[1:]),
            upsample_factor=4,
            return_error=False,
            )

shifts = np.apply_along_axis(func1d=infer_1, axis=1,
        arr=data.reshape((data.shape[0], -1)))

# calibrate images back by inferred shift
new_data = np.empty_like(data)
for i in range(new_data.shape[0]):
    img = data[i, ...]
    dx, dy = shifts[i, [1, 0]]
    tform = transform.AffineTransform(translation=(dx, dy))
    new_img = transform.warp(img, tform.inverse, preserve_range=True)
    new_data[i, ...] = new_img.astype(new_data.dtype)

# assess calibration result
new_shifts = np.apply_along_axis(func1d=infer_1, axis=1,
        arr=new_data.reshape((data.shape[0], -1)))

# visual check
fig, axes = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True)
plot_param = dict(cmap="coolwarm", s=5)

d = stats.gaussian_kde(shifts.T)(shifts.T)
sk = np.argsort(d)
axes[0].scatter(shifts[sk, 0], shifts[sk, 1], c=d[sk], **plot_param)
axes[0].set_title("before")

d = stats.gaussian_kde(new_shifts.T)(new_shifts.T)
sk = np.argsort(d)
axes[1].scatter(new_shifts[sk, 0], new_shifts[sk, 1], c=d[sk], **plot_param)
axes[1].set_title("after")

plt.tight_layout()
plt.show()
