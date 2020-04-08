import os
import sys

code_folderpath = os.path.expanduser('~/polartk/code/')
sys.path.append(code_folderpath)

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

    rgb_xy = rgb / rgb.max(axis=(0,1), keepdims=True) # normalize per channel
    gs_xy = rgb.max(axis=2)

    # convert to polar coord
    out_dict = polartk.xy2rt(images=[rgb_xy[..., ch]\
            for ch in range(rgb_xy.shape[2])])
    rgb_polar = np.stack(out_dict['image_rt_list'], axis=-1)

    out_dict = polartk.xy2rt(images=[gs_xy])
    gs_polar = out_dict['image_rt_list'][0]

    return rgb_xy, rgb_polar, gs_xy, gs_polar

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

    rgb_xy = rgb / rgb.max(axis=(0,1), keepdims=True) # normalize per channel
    gs_xy = rgb.max(axis=2)

    # convert to polar coord
    out_dict = polartk.xy2rt(images=[rgb_xy[..., ch]\
            for ch in range(rgb_xy.shape[2])])
    rgb_polar = np.stack(out_dict['image_rt_list'], axis=-1)

    out_dict = polartk.xy2rt(images=[gs_xy])
    gs_polar = out_dict['image_rt_list'][0]

    return rgb_xy, rgb_polar, gs_xy, gs_polar

if __name__ == '__main__':
    # load data
    demo1_rgb_xy, demo1_rgb_rt, demo1_gs_xy, demo1_gs_rt = demo1()
    demo2_rgb_xy, demo2_rgb_rt, demo2_gs_xy, demo2_gs_rt = demo2()

    demo1_cc = np.corrcoef(demo1_gs_rt)
    demo2_cc = np.corrcoef(demo2_gs_rt)

    # save to disk
    np.save('demo1_gs_rt.npy', demo1_gs_rt)
    np.save('demo2_gs_rt.npy', demo2_gs_rt)
    np.save('demo1_cc.npy', demo1_cc)
    np.save('demo2_cc.npy', demo2_cc)

    # plot
    fig, axes = plt.subplots(ncols=5, nrows=2, figsize=(10, 4),
            sharex=True, sharey=True)
    fs = 10

    axes[0, 0].imshow(demo1_rgb_xy)
    axes[0, 1].imshow(demo1_rgb_rt)
    axes[0, 2].imshow(demo1_gs_xy, cmap='gray')
    axes[0, 3].imshow(demo1_gs_rt, cmap='gray')
    axes[0, 4].imshow(demo1_cc, cmap='coolwarm', vmin=-1, vmax=1)

    axes[1, 0].imshow(demo2_rgb_xy)
    axes[1, 1].imshow(demo2_rgb_rt)
    axes[1, 2].imshow(demo2_gs_xy, cmap='gray')
    axes[1, 3].imshow(demo2_gs_rt, cmap='gray')
    axes[1, 4].imshow(demo2_cc, cmap='coolwarm', vmin=-1, vmax=1)

    axes[0, 0].set_title('Euclidean coordinate\n(RGB)', fontsize=fs, ma='center')
    axes[0, 1].set_title('Polar coordinate\n(RGB)', fontsize=fs, ma='center')
    axes[0, 2].set_title('Euclidean coordinate\n(gray scale)', fontsize=fs, ma='center')
    axes[0, 3].set_title('Polar coordinate\n(gray scale)', fontsize=fs, ma='center')
    axes[0, 4].set_title('Radial cross correlation', fontsize=fs, ma='center')

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.savefig(os.path.expanduser('~/polartk/figures/'\
            'demo_polar_coordinate_cross_correlation.png'), dpi=600)
    plt.show()
