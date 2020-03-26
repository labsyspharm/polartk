import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

import celltk

def eval_grid(distr_fn, im):
    x, y = np.meshgrid(range(im.shape[0]), range(im.shape[1]))
    xy = np.stack([x.flatten(), y.flatten()], axis=-1)
    z = distr_fn.pdf(xy).reshape(im.shape)
    return z

def make_polarized(image, centroid, cell_width, angle):
    # make positive part
    c1 = centroid
    w1 = cell_width
    fn1 = stats.multivariate_normal(mean=c1, cov=[w1, w1])
    im1 = eval_grid(fn1, image)

    # make negative part
    r = 1.3
    c2 = (centroid[0] + r * np.cos(np.pi - angle),
            centroid[1] + r * np.sin(np.pi - angle))
    w2 = cell_width * 0.55
    fn2 = stats.multivariate_normal(mean=c2, cov=[w2, w2])
    im2 = eval_grid(fn2, image)

    # set to only positive
    im = im1 - im2
    im[im < 0] = 0
    return im

if __name__ == '__main__':
    # image
    image_width = 30
    image = np.zeros((image_width, image_width))

    # centered cell
    center_cell = make_polarized(image, centroid=(14.5, 14.5), cell_width=10,
            angle=np.radians(135))

    # neighbor cells
    sat_cell_1 = make_polarized(image, centroid=(10, 1), cell_width=10,
            angle=np.radians(-60))
    sat_cell_2 = make_polarized(image, centroid=(1, 10), cell_width=10,
            angle=np.radians(-30))
    sat_cell_3 = make_polarized(image, centroid=(28, 28), cell_width=10,
            angle=np.radians(135))

    # make RGB image
    sat_cell = [sat_cell_1, sat_cell_2, sat_cell_3]
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
    plt.savefig('demo_2.png')
    plt.show()
