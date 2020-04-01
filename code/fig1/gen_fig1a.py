import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colorbar as mcolorbar
import matplotlib.colors as mcolors
import matplotlib.cm as mcolormap

from scipy.ndimage import morphology

if __name__ == '__main__':
    # paths
    xydata_filepath = './job_1497.npy'
    rtdata_filepath = './output_job_1497.npy'
    output_filepath = os.path.expanduser('~/polartk/figures/fig_1a.png')

    # load data
    label_xy = np.load(xydata_filepath)[..., 0]
    label_rt = np.load(rtdata_filepath)[..., 0]

    # calculate coordinates
    x, y = np.meshgrid(range(label_xy.shape[0]), range(label_xy.shape[1]),
            indexing='ij')
    xc, yc = np.argwhere(label_xy == 2).mean(axis=0)

    r_nuclei = morphology.distance_transform_edt(label_xy == 2)
    r_nuclei = r_nuclei.max() - r_nuclei
    r_cell = morphology.distance_transform_edt(label_xy < 2)
    r_background = morphology.distance_transform_edt(label_xy == 0)
    r = r_nuclei + r_cell + r_background

    t = np.arctan2(x-xc, y-yc)

    r_grid, t_grid = np.meshgrid(
            np.linspace(start=0, stop=r.max(), num=label_xy.shape[0]),
            np.linspace(start=-np.pi, stop=np.pi, num=label_xy.shape[1], endpoint=False),
            indexing='ij')

    # plot
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(6, 4),
            sharex=True, sharey=True)
    axes[0, 0].imshow(label_xy, cmap='tab10')
    axes[0, 0].scatter([yc], [xc], c='r', marker='o', s=5)
    axes[1, 0].imshow(label_rt, cmap='tab10')
    axes[0, 1].imshow(r, cmap='coolwarm')
    axes[0, 2].imshow(t, cmap='coolwarm')
    axes[1, 1].imshow(r_grid, cmap='coolwarm')
    axes[1, 2].imshow(t_grid, cmap='coolwarm')

    axes[0, 0].set_title('image/mask')
    axes[0, 1].set_title('radius')
    axes[0, 2].set_title('angle')

    axes[0, 0].set_ylabel('Euclidean\ncoordinate')
    axes[1, 0].set_ylabel('Polar\ncoordinate')

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    # custom legend
    legend_elements = [
            mlines.Line2D([0], [0], marker='o', color='w',
                markerfacecolor='r', markeredgecolor='r',
                markersize=2, label='cell centroid'),
            mpatches.Patch(facecolor='tab:cyan', edgecolor='tab:cyan',
                label='nuclei'),
            mpatches.Patch(facecolor='tab:brown', edgecolor='tab:brown',
                label='cytoplasm'),
            mpatches.Patch(facecolor='tab:blue', edgecolor='tab:blue',
                label='environment'),
            ]
    plt.figlegend(handles=legend_elements, ncol=2, bbox_to_anchor=(0.07, 0),
            loc='lower left', fontsize=6)

    # custom colorbar
    ax_colorbar = fig.add_axes([0.405, 0.048, 0.25, 0.03])
    cbar = mcolorbar.ColorbarBase(ax_colorbar, cmap=mcolormap.coolwarm,
            orientation='horizontal',
            norm=mcolors.Normalize(vmin=0, vmax=1))
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['low', 'high'])
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('numerical value', fontsize=6)
    cbar.ax.xaxis.set_label_coords(0.5, -0.5)

    # layout and show
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_filepath, dpi=600)
    plt.show()
