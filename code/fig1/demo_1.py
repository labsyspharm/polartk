import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

import celltk

def eval_grid(distr_fn, im):
    x, y = np.meshgrid(range(im.shape[0]), range(im.shape[1]))
    xy = np.stack([x.flatten(), y.flatten()], axis=-1)
    z = distr_fn.pdf(xy).reshape(im.shape)
    return z

if __name__ == '__main__':
    # image
    image_width = 30
    image = np.zeros((image_width, image_width))

    # centered cell
    centroid = ((image_width-1)/2, (image_width-1)/2)
    cell_width = 10
    distr_fn = stats.multivariate_normal(mean=centroid,
            cov=[cell_width, cell_width])
    center_cell = eval_grid(distr_fn, image)

    # neighbor cells
    center = [(14.5, 0), (0, 14.5), (image_width-2, image_width-2)]
    sat_cell = []
    for c in center:
        distr_fn = stats.multivariate_normal(mean=c,
                cov=[cell_width, cell_width])
        cell = eval_grid(distr_fn, image)
        sat_cell.append(cell)

    # make RGB image
    rgb = np.stack([image]*3, axis=-1)
    for ch in range(rgb.shape[2]):
        rgb[..., ch] += center_cell
        rgb[..., ch] += sat_cell[ch]

    rgb /= rgb.max(axis=(0,1), keepdims=True)

    # convert to polar coord
    rgb_polar = np.zeros_like(rgb)
    for ch in range(rgb.shape[2]):
        rgb_polar[..., ch] = celltk.xy2rt(rgb[..., ch])

    plt.subplot(121)
    plt.imshow(rgb)
    plt.title('in xy-coordinate')
    plt.xticks([]); plt.yticks([])

    plt.subplot(122)
    plt.imshow(rgb_polar)
    plt.title('in polar coordinate')
    plt.xticks([]); plt.yticks([])

    plt.suptitle('Synthetic cell image')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('demo_1.png')
    plt.show()
