import os
import sys
code_folderpath = os.path.expanduser('~/polartk/code/')
sys.path.append(code_folderpath)

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import morphology
from sklearn import neighbors
from skimage import draw

import polartk

def xy2rt_mask(mask_xy):
    x, y = np.meshgrid(range(mask_xy.shape[0]), range(mask_xy.shape[1]),
            indexing='ij')

    r_nuclei = morphology.distance_transform_edt(mask_xy == 2)
    r_nuclei = r_nuclei.max() - r_nuclei
    r_cell = morphology.distance_transform_edt(mask_xy < 2)
    r_background = morphology.distance_transform_edt(mask_xy == 0)
    r = r_nuclei + r_cell + r_background

    xc, yc = np.argwhere(mask_xy).mean(axis=0)
    t = np.arctan2(x-xc, y-yc)

    rg, tg = np.meshgrid(
            np.linspace(0, r.max(), num=mask_xy.shape[0]),
            np.linspace(-np.pi, np.pi, num=mask_xy.shape[1], endpoint=False),
            indexing='ij')

    rt = np.stack([r.flatten(), t.flatten()], axis=-1)
    rtg = np.stack([rg.flatten(), tg.flatten()], axis=-1)
    model = neighbors.KNeighborsClassifier(n_neighbors=1, metric=polartk.polar_dist)
    model.fit(rt, mask_xy.flatten())
    mask_rt = model.predict(rtg).reshape(mask_xy.shape)
    return r, mask_rt

if __name__ == '__main__':
    # params
    tile_shape = (30, 30)
    shape_dict = {}
    output_filepath = os.path.expanduser('~/polartk/figures/fig_s3.png')

    # coordinates
    x, y = np.meshgrid(range(tile_shape[0]), range(tile_shape[1]),
            indexing='ij')
    xc, yc = np.mean(x), np.mean(y)
    r = np.sqrt((x-xc)**2 + (y-yc)**2)

    # define circles
    inner_circle = r < 4
    outer_circle = r < 8

    # define triangles
    vertex_angles = (180, 180+120, 180-120) # unit: degrees
    inner_size = 6
    outer_size = 12
    inner_triangle = draw.polygon(
            r=[xc + inner_size * np.cos(np.radians(a)) for a in vertex_angles],
            c=[yc + inner_size * np.sin(np.radians(a)) for a in vertex_angles],
            shape=tile_shape)
    outer_triangle = draw.polygon(
            r=[xc + outer_size * np.cos(np.radians(a)) for a in vertex_angles],
            c=[yc + outer_size * np.sin(np.radians(a)) for a in vertex_angles],
            shape=tile_shape)

    # circle + circle
    m = np.zeros(tile_shape)
    m[outer_circle] = 1
    m[inner_circle] = 2
    shape_dict['cc'] = (m,)

    # circle + triangle
    m = np.zeros(tile_shape)
    m[outer_triangle[0], outer_triangle[1]] = 1
    m[inner_circle] = 2
    shape_dict['ct'] = (m,)

    # triangle + circle
    m = np.zeros(tile_shape)
    m[outer_circle] = 1
    m[inner_triangle[0], inner_triangle[1]] = 2
    shape_dict['tc'] = (m,)

    # triangle + triangle
    m = np.zeros(tile_shape)
    m[outer_triangle[0], outer_triangle[1]] = 1
    m[inner_triangle[0], inner_triangle[1]] = 2
    shape_dict['tt'] = (m,)

    # run transformation
    for key in shape_dict:
        m = shape_dict[key][0]
        shape_dict[key] = (m,) + xy2rt_mask(m)

    fig, axes = plt.subplots(ncols=3, nrows=len(shape_dict), sharex=True, sharey=True,
            figsize=(3, 4))
    fs = 6

    for row, key in enumerate(shape_dict):
        m_xy, r, m_rt = shape_dict[key]
        axes[row, 0].imshow(m_xy, cmap='tab10')
        axes[row, 1].imshow(r, cmap='coolwarm')
        axes[row, 2].imshow(m_rt, cmap='tab10')

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_title('Euclidean', fontsize=fs)
    axes[0, 1].set_title('radius', fontsize=fs)
    axes[0, 2].set_title('polar', fontsize=fs)

    fig.text(0.05, 0.45, 'Scenarios', ha='center', fontsize=fs,
            rotation='vertical')

    fig.tight_layout(rect=[0.05, 0, 1, 1])
    plt.savefig(output_filepath, dpi=600)
    plt.show()
