import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

if __name__ == '__main__':
    # paths
    data_folderpath = os.path.expanduser('~/polartk/code/fig2/scenario_data')
    output_filepath_dict = {key: os.path.expanduser('~/polartk/figures/fig_s1{}.png'\
            .format(key)) for key in ['a', 'b', 'c', 'd']}

    ######### radial perturbation #########
    # load data
    xydata_dict = {}
    rtdata_dict = {}
    for filename in os.listdir(data_folderpath):
        if os.path.splitext(filename)[1] == '.npy' and\
                filename.startswith('radial_'):
            key = os.path.splitext(filename)[0][len('radial_'):]
            filepath = os.path.join(data_folderpath, filename)
            xydata_dict[key] = np.load(filepath)
            filepath = os.path.join(data_folderpath, 'output_' + filename)
            rtdata_dict[key] = np.load(filepath)

    # plot euclidean coordinate
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(5, 5))
    fs = 6

    scenario_list = ['e', 'x', 'd']
    for i, nuclei_scenario in enumerate(scenario_list):
        for j, cell_scenario in enumerate(scenario_list):
            scenario = 'n{}c{}'.format(nuclei_scenario, cell_scenario)
            data = xydata_dict[scenario][..., 0]
            axes[i, j].imshow(data, cmap='tab10')
            xc, yc = np.argwhere(data == 2).mean(axis=0)
            axes[i, j].scatter([yc], [xc], c='r', marker='o', s=2)
    
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_ylabel('erosion', fontsize=fs)
    axes[1, 0].set_ylabel('no change', fontsize=fs)
    axes[2, 0].set_ylabel('dilation', fontsize=fs)

    axes[2, 0].set_xlabel('erosion', fontsize=fs)
    axes[2, 1].set_xlabel('no change', fontsize=fs)
    axes[2, 2].set_xlabel('dilation', fontsize=fs)

    # custom legend
    legend_elements = [
            mlines.Line2D([0], [0], color='w', marker='o',
                markeredgecolor='r', markerfacecolor='r',
                markersize=5, label='cell centroid'),
            mpatches.Patch(facecolor='tab:cyan', edgecolor='tab:cyan',
                label='nuclei'),
            mpatches.Patch(facecolor='tab:brown', edgecolor='tab:brown',
                label='cytoplasm'),
            mpatches.Patch(facecolor='tab:blue', edgecolor='tab:blue',
                label='environment'),
            ]
    plt.figlegend(handles=legend_elements, ncol=4, bbox_to_anchor=(0.55, 0.93),
            loc='center', fontsize=6)

    fig.text(0.55, 0.02, 'cytoplasm mask', ha='center', fontsize=fs)
    fig.text(0.05, 0.45, 'nuclei mask', ha='center', fontsize=fs,
            rotation='vertical')
    fig.suptitle('Euclidean coordinate', fontsize=fs, y=0.98)
    fig.tight_layout(rect=[0.05, 0.02, 1, 0.93])
    plt.savefig(output_filepath_dict['a'], dpi=600)
    plt.show()
    plt.close()

    # plot polar coordinate
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(5, 5))
    fs = 6

    scenario_list = ['e', 'x', 'd']
    for i, nuclei_scenario in enumerate(scenario_list):
        for j, cell_scenario in enumerate(scenario_list):
            scenario = 'n{}c{}'.format(nuclei_scenario, cell_scenario)
            data = rtdata_dict[scenario][..., 0]
            axes[i, j].imshow(data, cmap='tab10')
    
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_ylabel('erosion', fontsize=fs)
    axes[1, 0].set_ylabel('no change', fontsize=fs)
    axes[2, 0].set_ylabel('dilation', fontsize=fs)

    axes[2, 0].set_xlabel('erosion', fontsize=fs)
    axes[2, 1].set_xlabel('no change', fontsize=fs)
    axes[2, 2].set_xlabel('dilation', fontsize=fs)

    # custom legend
    legend_elements = [
            mpatches.Patch(facecolor='tab:cyan', edgecolor='tab:cyan',
                label='nuclei'),
            mpatches.Patch(facecolor='tab:brown', edgecolor='tab:brown',
                label='cytoplasm'),
            mpatches.Patch(facecolor='tab:blue', edgecolor='tab:blue',
                label='environment'),
            ]
    plt.figlegend(handles=legend_elements, ncol=3, bbox_to_anchor=(0.55, 0.93),
            loc='center', fontsize=6)

    fig.text(0.55, 0.02, 'cytoplasm mask', ha='center', fontsize=fs)
    fig.text(0.05, 0.45, 'nuclei mask', ha='center', fontsize=fs,
            rotation='vertical')
    fig.suptitle('polar coordinate', fontsize=fs, y=0.98)
    fig.tight_layout(rect=[0.05, 0.02, 1, 0.93])
    plt.savefig(output_filepath_dict['b'], dpi=600)
    plt.show()
    plt.close()

    ######### translational perturbation #########
    # load data
    xydata_dict = {}
    rtdata_dict = {}
    for filename in os.listdir(data_folderpath):
        if os.path.splitext(filename)[1] == '.npy' and\
                filename.startswith('translational_'):
            key = os.path.splitext(filename)[0][len('translational_'):]
            filepath = os.path.join(data_folderpath, filename)
            xydata_dict[key] = np.load(filepath)
            filepath = os.path.join(data_folderpath, 'output_' + filename)
            rtdata_dict[key] = np.load(filepath)

    # plot euclidean coordinate
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(5, 5))
    fs = 6

    scenario_list = ['-1', '0', '1']
    for i, shift_x in enumerate(scenario_list):
        for j, shift_y in enumerate(scenario_list):
            scenario = 'c{}{}'.format(shift_x, shift_y)
            data = xydata_dict[scenario][..., 0]
            axes[i, j].imshow(data, cmap='tab10')
            xc, yc = np.argwhere(data == 2).mean(axis=0)
            axes[i, j].scatter([yc], [xc], c='r', marker='o', s=2)
    
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_ylabel('shift -1', fontsize=fs)
    axes[1, 0].set_ylabel('no change', fontsize=fs)
    axes[2, 0].set_ylabel('shift +1', fontsize=fs)

    axes[2, 0].set_xlabel('shift -1', fontsize=fs)
    axes[2, 1].set_xlabel('no change', fontsize=fs)
    axes[2, 2].set_xlabel('shift +1', fontsize=fs)

    # custom legend
    legend_elements = [
            mlines.Line2D([0], [0], color='w', marker='o',
                markeredgecolor='r', markerfacecolor='r',
                markersize=5, label='cell centroid'),
            mpatches.Patch(facecolor='tab:cyan', edgecolor='tab:cyan',
                label='nuclei'),
            mpatches.Patch(facecolor='tab:brown', edgecolor='tab:brown',
                label='cytoplasm'),
            mpatches.Patch(facecolor='tab:blue', edgecolor='tab:blue',
                label='environment'),
            ]
    plt.figlegend(handles=legend_elements, ncol=4, bbox_to_anchor=(0.55, 0.93),
            loc='center', fontsize=6)

    fig.text(0.55, 0.02, 'shift in y-direction', ha='center', fontsize=fs)
    fig.text(0.05, 0.45, 'shift in x-direction', ha='center', fontsize=fs,
            rotation='vertical')
    fig.suptitle('Euclidean coordinate', fontsize=fs, y=0.98)
    fig.tight_layout(rect=[0.05, 0.02, 1, 0.93])
    plt.savefig(output_filepath_dict['c'], dpi=600)
    plt.show()
    plt.close()

    # plot polar coordinate
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(5, 5))
    fs = 6

    scenario_list = ['-1', '0', '1']
    for i, shift_x in enumerate(scenario_list):
        for j, shift_y in enumerate(scenario_list):
            scenario = 'c{}{}'.format(shift_x, shift_y)
            data = rtdata_dict[scenario][..., 0]
            axes[i, j].imshow(data, cmap='tab10')
    
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_ylabel('shift -1', fontsize=fs)
    axes[1, 0].set_ylabel('no change', fontsize=fs)
    axes[2, 0].set_ylabel('shift +1', fontsize=fs)

    axes[2, 0].set_xlabel('shift -1', fontsize=fs)
    axes[2, 1].set_xlabel('no change', fontsize=fs)
    axes[2, 2].set_xlabel('shift +1', fontsize=fs)

    # custom legend
    legend_elements = [
            mpatches.Patch(facecolor='tab:cyan', edgecolor='tab:cyan',
                label='nuclei'),
            mpatches.Patch(facecolor='tab:brown', edgecolor='tab:brown',
                label='cytoplasm'),
            mpatches.Patch(facecolor='tab:blue', edgecolor='tab:blue',
                label='environment'),
            ]
    plt.figlegend(handles=legend_elements, ncol=3, bbox_to_anchor=(0.55, 0.93),
            loc='center', fontsize=6)

    fig.text(0.55, 0.02, 'shift in y-direction', ha='center', fontsize=fs)
    fig.text(0.05, 0.45, 'shift in x-direction', ha='center', fontsize=fs,
            rotation='vertical')
    fig.suptitle('polar coordinate', fontsize=fs, y=0.98)
    fig.tight_layout(rect=[0.05, 0.02, 1, 0.93])
    plt.savefig(output_filepath_dict['d'], dpi=600)
    plt.show()
    plt.close()
