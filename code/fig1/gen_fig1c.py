import os

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # paths
    marker_filepath = os.path.expanduser('~/polar_data/data/markers.csv')
    xydata_filepath = 'job_1497.npy'
    rtdata_filepath = 'output_job_1497.npy'
    output_filepath = os.path.expanduser('~/polartk/figures/fig_1c.png')

    # get marker list
    with open(marker_filepath, 'r') as infile:
        marker_list = [line.strip() for line in infile.readlines()]
    all_index = list(range(len(marker_list)))
    dna_index = list(range(4, len(marker_list), 4))
    background_index = [1, 2, 3]
    marker_index = set(all_index).difference(set(dna_index))\
            .difference(set(background_index))
    marker_index = list(marker_index)
    index_dna1 = marker_index.index(marker_list.index('DNA1'))
    index_pd1 = marker_index.index(marker_list.index('PD1'))

    # load data
    image_xy = np.load(xydata_filepath)
    image_rt = np.load(rtdata_filepath)
    dna1_xy = image_xy[..., 1+index_dna1]
    pd1_xy = image_xy[..., 1+index_pd1]
    dna1_rt = image_rt[..., 1+index_dna1]
    pd1_rt = image_rt[..., 1+index_pd1]

    # plot
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(4,4),
            sharex=True, sharey=True)

    p_dna1 = dict(cmap='gray', vmin=min(dna1_xy.min(), dna1_rt.min()),
            vmax=max(dna1_xy.max(), dna1_rt.max()))
    axes[0, 0].imshow(dna1_xy, **p_dna1)
    axes[0, 1].imshow(dna1_rt, **p_dna1)

    p_pd1 = dict(cmap='gray', vmin=min(pd1_xy.min(), pd1_rt.min()),
            vmax=max(pd1_xy.max(), pd1_rt.max()))
    axes[1, 0].imshow(pd1_xy, **p_pd1)
    axes[1, 1].imshow(pd1_rt, **p_pd1)

    axes[0, 0].set_title('Euclidean\ncoordinate')
    axes[0, 1].set_title('Polar\ncoordinate')
    axes[0, 0].set_ylabel('DNA')
    axes[1, 0].set_ylabel('PD1')

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.savefig(output_filepath, dpi=600)
    plt.show()
