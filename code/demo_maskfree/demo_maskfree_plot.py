import os

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # path
    output_filepath = os.path.expanduser('~/polartk/figures/demo_maskfree_transform.png')

    # load data
    im_xy = np.load('im_xy.npy')
    im_rt_std = np.load('im_rt_std.npy')
    im_rt_mask = np.load('im_rt_mask.npy')
    im_rt_maskfree = np.load('im_rt_maskfree.npy')
    im_list = [im_xy, im_rt_std, im_rt_mask, im_rt_maskfree]

    # plot
    ch_DNA1, ch_PD1 = 0, 12
    fs = 12
    fig, axes = plt.subplots(ncols=4, nrows=2, sharex=True, sharey=True,
            figsize=(12, 6))

    for row_index, ch in enumerate([ch_DNA1, ch_PD1]):
        params = dict(cmap='gray',
                vmin=min([im[..., ch].min() for im in im_list]),
                vmax=max([im[..., ch].max() for im in im_list]),
                )

        axes[row_index, 0].imshow(im_xy[..., ch], **params)
        axes[row_index, 1].imshow(im_rt_std[..., ch], **params)
        axes[row_index, 2].imshow(im_rt_mask[..., ch], **params)
        axes[row_index, 3].imshow(im_rt_maskfree[..., ch], **params)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_title('Euclidean coordinate', fontsize=fs)
    axes[0, 1].set_title('Previously, no mask', fontsize=fs)
    axes[0, 2].set_title('Previously, with mask', fontsize=fs)
    axes[0, 3].set_title('New, no mask', fontsize=fs)

    axes[0, 0].set_ylabel('DNA1', fontsize=fs)
    axes[1, 0].set_ylabel('PD1', fontsize=fs)

    fig.tight_layout()
    plt.savefig(output_filepath, dpi=600)
    plt.show()

