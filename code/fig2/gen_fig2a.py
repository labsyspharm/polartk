import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colorbar as mcolorbar
import matplotlib.cm as mcolormap
import matplotlib.colors as mcolors

if __name__ == '__main__':
    # paths
    data_folderpath = '.'
    output_filepath = os.path.expanduser('~/polartk/figures/fig_2a.png')

    # load data
    xydata_dict = {}
    rtdata_dict = {}
    for filename in os.listdir(data_folderpath):
        if os.path.splitext(filename)[1] != '.npy':
            continue
        filepath = os.path.join(data_folderpath, filename)
        scenario = os.path.splitext(filename)[0].split('_')[-1]
        data = np.load(filepath)
        if filename.startswith('scenario'): # xy
            xydata_dict[scenario] = data
        elif filename.startswith('output'): # rt
            rtdata_dict[scenario] = data

    # plot
    fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(8, 3))

    original_pd1 = xydata_dict['nxcx'][..., 1]
    axes[0].imshow(original_pd1, cmap='gray')
    axes[0].set_title('image (Euclidean)', fontsize=8)

    original_label_xy = xydata_dict['nxcx'][..., 0]
    axes[1].imshow(original_label_xy, cmap='tab10')
    axes[1].set_title('mask (Euclidean)', fontsize=8)

    original_label_rt = rtdata_dict['nxcx'][..., 0]
    axes[2].imshow(original_label_rt, cmap='tab10')
    axes[2].set_title('mask (polar)', fontsize=8)

    for ax in axes[0:3]:
        ax.set_xticks([])
        ax.set_yticks([])

    tdist_list = []
    for rtdata in rtdata_dict.values():
        tdist = rtdata[..., 1].sum(axis=0) # (R, T) -> (T,)
        tdist /= tdist.sum() # normalize total expression abundance
        tdist_list.append(tdist)

    t_grid = np.linspace(-np.pi, np.pi, num=tdist_list[0].shape[0], endpoint=False)
    t_grid = np.degrees(t_grid)

    for tdist in tdist_list:
        axes[3].plot(t_grid, tdist, 'k-', alpha=0.1)

    axes[3].set_yticks([])
    axes[3].set_xticks(np.linspace(-180, 180, num=3))
    axes[3].set_xticklabels(np.linspace(-180, 180, num=3).astype(int), fontsize=8)
    axes[3].set_title('angular distributions', fontsize=8)
    axes[3].set_xlabel('angle (degrees)', fontsize=8)
    axes[3].set_ylabel('normalized intensity', fontsize=8)

    asp = np.diff(axes[3].get_xlim())[0] / np.diff(axes[3].get_ylim())[0]
    axes[3].set_aspect(asp)

    # custom legend
    legend_elements = [
            mlines.Line2D([0], [0], color='black', alpha=0.1,
                label='perturbations (n={})'.format(len(tdist_list))),
            ]
    axes[3].legend(handles=legend_elements, fontsize=6)

    # custom legend
    legend_elements = [
            mpatches.Patch(facecolor='tab:cyan', edgecolor='tab:cyan',
                label='nuclei'),
            mpatches.Patch(facecolor='tab:brown', edgecolor='tab:brown',
                label='cytoplasm'),
            mpatches.Patch(facecolor='tab:blue', edgecolor='tab:blue',
                label='environment'),
            ]
    plt.figlegend(handles=legend_elements, ncol=3, bbox_to_anchor=(0.5, 0.11),
            loc='lower center', fontsize=6)

    # custom colorbar
    ax_colorbar = fig.add_axes([0.025, 0.15, 0.205, 0.03])
    cbar = mcolorbar.ColorbarBase(ax_colorbar, cmap=mcolormap.gray,
            orientation='horizontal',
            norm=mcolors.Normalize(vmin=0, vmax=1))
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['low', 'high'])
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('intensity', fontsize=6)
    cbar.ax.xaxis.set_label_coords(0.5, -0.5)

    fig.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(output_filepath, dpi=600)
    plt.show()
