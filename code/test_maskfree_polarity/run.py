import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

import polartk

def eval_grid(distr_fn, shape):
    x, y = np.meshgrid(range(shape[0]), range(shape[1]), indexing='ij')
    xy = np.stack([x.flatten(), y.flatten()], axis=-1)
    z = distr_fn.pdf(xy).reshape(shape)
    return z

def eval_grid_polarized(shape, centroid, cell_width, angle):
    # make positive part
    c1 = centroid
    w1 = cell_width
    fn1 = stats.multivariate_normal(mean=c1, cov=[w1, w1])
    im1 = eval_grid(fn1, shape)

    # make negative part
    r = 1.3
    c2 = (centroid[0] + r * np.cos(np.pi - angle),
            centroid[1] + r * np.sin(np.pi - angle))
    w2 = cell_width * 0.55
    fn2 = stats.multivariate_normal(mean=c2, cov=[w2, w2])
    im2 = eval_grid(fn2, shape)

    # set to only positive
    im = im1 - im2
    im[im < 0] = 0
    return im

def demo1():
    # params
    tile_shape = (30, 30)

    # centered cell
    centroid = ((tile_shape[0]-1)/2, (tile_shape[1]-1)/2)
    cell_width = 10
    distr_fn = stats.multivariate_normal(mean=centroid,
            cov=(cell_width, cell_width))
    center_cell = eval_grid(distr_fn, tile_shape)

    # neighbor cells
    center = [(14.5, 0), (0, 14.5), (tile_shape[0]-2, tile_shape[1]-2)]
    sat_cell = []
    for c in center:
        distr_fn = stats.multivariate_normal(mean=c,
                cov=(cell_width, cell_width))
        cell = eval_grid(distr_fn, tile_shape)
        sat_cell.append(cell)

    # make RGB image
    rgb = np.zeros(tile_shape + (3,))
    for ch in range(rgb.shape[2]):
        rgb[..., ch] += center_cell
        rgb[..., ch] += sat_cell[ch]

    rgb /= rgb.max(axis=(0,1), keepdims=True) # normalize per channel

    # convert to polar coord
    rgb_polar = np.zeros_like(rgb)
    for ch in range(rgb.shape[2]):
        _, _, image_rt = polartk.xy2rt(rgb[..., ch])
        rgb_polar[..., ch] = image_rt

    return rgb, rgb_polar

def demo2():
    # params
    tile_shape = (30, 30)

    # centered cell
    center_cell = eval_grid_polarized(tile_shape, centroid=(14.5, 14.5),
            cell_width=10, angle=np.radians(135))

    # neighbor cells
    sat_cell_1 = eval_grid_polarized(tile_shape, centroid=(10, 1),
            cell_width=10, angle=np.radians(-60))
    sat_cell_2 = eval_grid_polarized(tile_shape, centroid=(1, 10),
            cell_width=10, angle=np.radians(-30))
    sat_cell_3 = eval_grid_polarized(tile_shape, centroid=(28, 28),
            cell_width=10, angle=np.radians(135))

    # make RGB image
    sat_cell = [sat_cell_1, sat_cell_2, sat_cell_3]
    rgb = np.zeros(tile_shape + (3,))
    for ch in range(rgb.shape[2]):
        rgb[..., ch] += center_cell
        rgb[..., ch] += sat_cell[ch]

    rgb /= rgb.max(axis=(0,1), keepdims=True) # normalize per channel

    # convert to polar coord
    rgb_polar = np.zeros_like(rgb)
    for ch in range(rgb.shape[2]):
        _, _, image_rt = polartk.xy2rt(rgb[..., ch])
        rgb_polar[..., ch] = image_rt

    return rgb, rgb_polar

if __name__ == '__main__':
    # load data
    demo1_outdict = demo1()
    demo2_outdict = demo2()

    demo1_cc = np.corrcoef(demo1_outdict)
    demo2_cc = np.corrcoef(demo2_rt)

    plt.subplot(121)
    plt.imshow(demo1_cc, cmap='coolwarm')
    plt.subplot(122)
    plt.imshow(demo2_cc, cmap='coolwarm')
    plt.show()
