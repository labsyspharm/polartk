import os
import sys
code_folderpath = os.path.expanduser('~/polartk/code/')
sys.path.append(code_folderpath)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import neighbors
from scipy import stats

import polartk

def tdist2rt(tdist, shape):
    image_rt = np.zeros(shape)
    # assume 8 pixels thick of nuclei, 8 pixels think of cytoplasm
    for r in range(8, 17): 
        image_rt[r, :] = tdist
    return image_rt

def rt2xy(image_rt):
    r, t = np.meshgrid(
            np.linspace(0, image_rt.shape[0]/2, num=image_rt.shape[0]),
            np.linspace(-np.pi, np.pi, num=image_rt.shape[1], endpoint=False),
            indexing='ij')
    x = r * np.cos(t)
    y = r * np.sin(t)
    xy = np.stack([x.flatten(), y.flatten()], axis=-1)

    xg, yg = np.meshgrid(np.arange(image_rt.shape[0], dtype=float),
            np.arange(image_rt.shape[1], dtype=float),
            indexing='ij')
    xyg = np.stack([xg.flatten(), yg.flatten()], axis=-1)
    xyg -= xyg.mean(axis=0, keepdims=True)

    model = neighbors.KNeighborsRegressor(n_neighbors=5)
    model.fit(xy, image_rt.flatten())
    image_xy = model.predict(xyg).reshape(image_rt.shape)
    return image_xy

if __name__ == '__main__':
    # path
    output_filepath = os.path.expanduser('~/polartk/figures/fig_2b.png')

    # prepare all peaks
    num_t = 30
    peak_dict = {}

    def fn(d):
        image_rt = tdist2rt(d, shape=(num_t, num_t))
        image_xy = rt2xy(image_rt)
        p = polartk.polarity(d)
        return (image_xy, p)

    def random_tdist():
        num_avg = 8 # assume 5 pixels thick of cytoplasm
        d = np.random.rand(num_avg, num_t)\
                .reshape((num_avg, num_t))\
                .mean(axis=0) # average over radii
        d /= d.sum()
        return d

    # reference
    d = stats.norm(loc=15, scale=6).pdf(range(num_t))
    d /= d.sum()
    peak_dict['reference'] = fn(d)

    # no polarization
    d = np.ones(num_t)
    d /= d.sum()
    peak_dict['no polarization'] = fn(d)

    # more polarization
    d = stats.norm(loc=15, scale=2.5).pdf(range(num_t))
    d /= d.sum()
    peak_dict['more polarization'] = fn(d)

    # rotated
    d = stats.norm(loc=15, scale=6).pdf(range(num_t))
    d = np.roll(d, 10)
    d /= d.sum()
    peak_dict['rotated'] = fn(d)

    # double polarization
    d = stats.norm(loc=5, scale=3).pdf(range(num_t))
    d += np.roll(d, 15)
    d /= d.sum()
    peak_dict['double polarization'] = fn(d)

    # multiple polarization
    d = np.zeros(num_t)
    d[0::4] = 1
    d[1::4] = 1
    d /= d.sum()
    peak_dict['multiple polarization'] = fn(d)

    # random
    d = random_tdist()
    peak_dict['random'] = fn(d)

    # many random for baseline
    num_sample = int(1e5)
    p_rand = np.zeros(num_sample)
    for i in range(num_sample):
        d_rand = random_tdist()
        p_rand[i] = polartk.polarity(d_rand)

    fig, axes = plt.subplots(ncols=4, nrows=2, figsize=(7, 4.5))
    axes = axes.flatten()

    angle = np.linspace(-np.pi, np.pi, num=num_t, endpoint=False)
    angle = np.degrees(angle)

    params = dict(cmap='gray',
            vmin=min([t[0].min() for t in peak_dict.values()]),
            vmax=max([t[0].max() for t in peak_dict.values()]),
            )

    fs = 8

    for i, (ax, key) in enumerate(zip(axes, peak_dict)):
        im, p = peak_dict[key]
        ax.imshow(im, **params)
        ax.set_title('({}) {}'.format(i+1, key), fontsize=fs)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[7].hist(p_rand, bins=20, density=True)

    key_list = list(peak_dict.keys())
    params = {'arrowprops': dict(arrowstyle='->'), 'fontsize': 6, 'ha': 'center'}
    for i in [0, 2]:
        key = key_list[i]
        _, p = peak_dict[key]
        axes[7].annotate(s='({})'.format(i+1), xy=(p, 1), xytext=(p, 10),
                **params)
    for i in [1, 3, 4, 5]:
        key = key_list[i]
        _, p = peak_dict[key]
        axes[7].annotate(s='({})'.format(i+1), xy=(p, 1), xytext=(p+0.1, 15),
                **params)
    for i in [6]:
        key = key_list[i]
        _, p = peak_dict[key]
        axes[7].annotate(s='({})'.format(i+1), xy=(p, 1), xytext=(p+0.2, 20),
                **params)

    axes[7].set_title('histogram', fontsize=fs)
    axes[7].legend(['{:.0f}k random\nsimulations'.format(num_sample/1e3)],
            fontsize=fs-2)
    axes[7].set_xlabel('polarity', fontsize=fs)
    axes[7].set_ylabel('count', fontsize=fs)
    axes[7].set_xticks([0, 1])
    axes[7].set_xticklabels([0, 1], fontsize=fs)
    asp = np.diff(axes[7].get_xlim())[0] / np.diff(axes[7].get_ylim())[0]
    axes[7].set_aspect(asp)
    axes[7].set_yticks([])

    fig.suptitle('benchmark of polarity metric\n'\
            'range from 0 (completely flat) to 1 (completely polarized)',
            y=0.98, fontsize=fs)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_filepath, dpi=600)
    plt.show()
