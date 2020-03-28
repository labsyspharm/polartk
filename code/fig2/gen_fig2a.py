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
        p = mpatches.FancyArrowPatch(posA=(0, 2), posB=(22, 2),
                mutation_scale=10,
                color='magenta', arrowstyle='<->')
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

if __name__ == '__main__':
    # paths
    data_folderpath = './scenario_data'
    output_filepath = os.path.expanduser('~/polartk/figures/fig_2a.png')

    # load reference data
    reference_filepath = os.path.join(data_folderpath, 'radial_nxcx.npy')
    reference = np.load(reference_filepath)
    ref_image, ref_label = reference[..., 1], reference[..., 0]

    # load angular distribution data
    radial_dict = {}
    translational_dict = {}
    full_tgrid = np.linspace(-np.pi, np.pi, num=ref_image.shape[1], endpoint=False)
    full_tgrid = np.degrees(full_tgrid)
    for name in os.listdir(data_folderpath):
        if os.path.splitext(name)[1] == '.npy' and name.startswith('output_'):
            filepath = os.path.join(data_folderpath, name)
            data = np.load(filepath)
            mask = data[..., 0] == 1 # cytoplasm
            image = data[..., 1]

            tdist = np.multiply(image, mask).sum(axis=0)
            valid_mask = mask.sum(axis=0) > 0
            tdist = tdist[valid_mask]
            tgrid = full_tgrid[valid_mask]

            tdist /= tdist.sum() # normalize total expression abundance
            if 'radial_' in name:
                key = os.path.splitext(name)[0][len('output_radial_'):]
                radial_dict[key] = (tgrid, tdist)
            elif 'translational' in name:
                key = os.path.splitext(name)[0][len('output_translational_'):]
                translational_dict[key] = (tgrid, tdist)

    # plot
    fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(5.5, 9))
    fs = 10
    arrow_color = 'black'

    axes[0, 0].imshow(ref_image, cmap='gray')
    axes[0, 0].set_title('image (Euclidean coordinate)', fontsize=fs)

    axes[0, 1].set_visible(False)

    axes[1, 0].imshow(ref_label, cmap='tab10')
    axes[1, 0].set_title('radial perturbations', fontsize=fs)
    axes[1, 0].annotate(s='', xy=(18, 16), xytext=(18+5, 16-5),
            arrowprops=dict(arrowstyle='<->', color=arrow_color))
    axes[1, 0].annotate(s='', xy=(18, 9), xytext=(18+5, 9-5),
            arrowprops=dict(arrowstyle='<->', color=arrow_color))

    axes[2, 0].imshow(ref_label, cmap='tab10')
    axes[2, 0].set_title('translational perturbations', fontsize=fs)
    xc, yc = 20.5, 6.5
    length = 5*np.sqrt(2)
    axes[2, 0].annotate(s='', xy=(xc-length/2, yc), xytext=(xc+length/2, yc),
            arrowprops=dict(arrowstyle='<->', color=arrow_color))
    axes[2, 0].annotate(s='', xy=(xc, yc-length/2), xytext=(xc, yc+length/2),
            arrowprops=dict(arrowstyle='<->', color=arrow_color))

    for ax in axes[:, 0].flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    for key in radial_dict:
        tgrid, tdist = radial_dict[key]
        axes[1, 1].plot(tgrid, tdist, 'k-', alpha=0.1)

    for key in translational_dict:
        tgrid, tdist = translational_dict[key]
        axes[2, 1].plot(tgrid, tdist, 'k-', alpha=0.1)

    n_list = [len(radial_dict), len(translational_dict)]
    for ax, n in zip(axes[1:, 1].flatten(), n_list):
        ax.set_yticks([])
        ax.set_xticks(np.linspace(-180, 180, num=3))
        ax.set_xticklabels(np.linspace(-180, 180, num=3).astype(int), fontsize=fs)
        ax.set_xlabel('angle (degrees)', fontsize=fs)
        ax.set_ylabel('normalized intensity', fontsize=fs)

        ax.annotate(s='', xy=(-20, 0.027), xytext=(-20, 0.027+0.025),
                arrowprops=dict(arrowstyle='<->', color=arrow_color))

        asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
        ax.set_aspect(asp)

        # custom legend
        legend_elements = [
                mlines.Line2D([0], [0], color='black', alpha=0.1,
                    label='perturbations\n(n={})'.format(n)),
                ]
        ax.legend(handles=legend_elements, fontsize=fs)

    # custom legend
    legend_elements = [
            mpatches.Patch(facecolor='tab:cyan', edgecolor='tab:cyan',
                label='nuclei'),
            mpatches.Patch(facecolor='tab:brown', edgecolor='tab:brown',
                label='cytoplasm'),
            mpatches.Patch(facecolor='tab:blue', edgecolor='tab:blue',
                label='environment'),
            mpatches.FancyArrowPatch(posA=(0, 0), posB=(5, 0), arrowstyle='<->',
                label='perturbation\ndirections', color=arrow_color),
            ]
    plt.figlegend(handles=legend_elements, ncol=1, bbox_to_anchor=(0.65, 0.95),
            loc='upper left', fontsize=fs, title='mask legend',
            handler_map={mpatches.FancyArrowPatch : HandlerArrow()},
            )

    # custom colorbar
    ax_colorbar = fig.add_axes([0.5, 0.7, 0.03, 0.27])
    cbar = mcolorbar.ColorbarBase(ax_colorbar, cmap=mcolormap.gray,
            orientation='vertical',
            norm=mcolors.Normalize(vmin=0, vmax=1))
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['low', 'high'])
    cbar.ax.tick_params(labelsize=fs)
    cbar.set_label('image intensity', fontsize=fs, rotation='vertical',
            ha='center', va='top')
    cbar.ax.yaxis.set_label_coords(2, 0.5)

    fig.tight_layout()
    plt.savefig(output_filepath, dpi=600)
    plt.show()
