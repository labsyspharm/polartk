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

# visual check
d = stats.gaussian_kde(shifts.T)(shifts.T)
sk = np.argsort(d)
plt.scatter(shifts[sk, 0], shifts[sk, 1], c=d[sk], cmap="coolwarm")
plt.show()
