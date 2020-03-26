import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colorbar as mcolorbar
import matplotlib.cm as mcolormap
import matplotlib.colors as mcolors

from matplotlib.legend_handler import HandlerPatch

class HandlerArrow(HandlerPatch):
    '''
    For showing an arrow in a custom legend.

    Source:
    https://stackoverflow.com/questions/60781312/plotting-arrow-in-front-of-legend-matplotlib
    '''
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width,
            height, fontsize, trans):
        p = mpatches.FancyArrowPatch(posA=(-9, 2), posB=(15, 2),
                mutation_scale=8,
                color='magenta', arrowstyle='<->')
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

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
    axes[1].annotate(s='', xy=(18, 16), xytext=(18+5, 16-5),
            arrowprops=dict(arrowstyle='<->', color='magenta'))
    axes[1].annotate(s='', xy=(18, 9), xytext=(18+5, 9-5),
            arrowprops=dict(arrowstyle='<->', color='magenta'))

    original_label_rt = rtdata_dict['nxcx'][..., 0]
    axes[2].imshow(original_label_rt, cmap='tab10')
    axes[2].set_title('mask (polar)', fontsize=8)
    axes[2].annotate(s='', xy=(13.5, 5), xytext=(13.5+0, 5+5*np.sqrt(2)),
            arrowprops=dict(arrowstyle='<->', color='magenta'))
    axes[2].annotate(s='', xy=(20, 0), xytext=(20+0, 0+5*np.sqrt(2)),
            arrowprops=dict(arrowstyle='<->', color='magenta'))

    for ax in axes[0:3]:
        ax.set_xticks([])
        ax.set_yticks([])

    tgrid_list = []
    tdist_list = []
    full_tgrid = np.linspace(-np.pi, np.pi, num=30, endpoint=False)
    full_tgrid = np.degrees(full_tgrid)

    for rtdata in rtdata_dict.values():
        cytoplasm_mask = rtdata[..., 0] == 1
        intensity = rtdata[..., 1]
        tdist = np.multiply(intensity, cytoplasm_mask).sum(axis=0)
        valid_mask = cytoplasm_mask.sum(axis=0) > 0
        tdist = tdist[valid_mask]
        tgrid = full_tgrid[valid_mask]

        tdist /= tdist.sum() # normalize total expression abundance

        tgrid_list.append(tgrid)
        tdist_list.append(tdist)

    for tgrid, tdist in zip(tgrid_list, tdist_list):
        axes[3].plot(tgrid, tdist, 'k-', alpha=0.1)

    axes[3].set_yticks([])
    axes[3].set_xticks(np.linspace(-180, 180, num=3))
    axes[3].set_xticklabels(np.linspace(-180, 180, num=3).astype(int), fontsize=8)
    axes[3].set_title('angular distributions', fontsize=8)
    axes[3].set_xlabel('angle (degrees)', fontsize=8)
    axes[3].set_ylabel('normalized intensity', fontsize=8)

    axes[3].annotate(s='', xy=(-20, 0.027), xytext=(-20, 0.027+0.025),
            arrowprops=dict(arrowstyle='<->', color='magenta'))

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
            mpatches.FancyArrowPatch(posA=(0, 0), posB=(5, 0), arrowstyle='<->',
                label='perturbation directions', color='magenta'),
            ]
    plt.figlegend(handles=legend_elements, ncol=4, bbox_to_anchor=(0.5, 0.11),
            loc='lower center', fontsize=6,
            handler_map={mpatches.FancyArrowPatch : HandlerArrow()},
            )

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
