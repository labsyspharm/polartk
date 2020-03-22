import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

if __name__ == '__main__':
    # paths
    data_folderpath = os.path.expanduser('~/polar_data')
    cc_filepath = os.path.join(data_folderpath, 'pd1_env_corrcoef.csv')
    marker_filepath = os.path.join(data_folderpath, 'data', 'markers.csv')
    output_fig_filepath = '../figures/pd1_correlation.png'
    
    # load data
    cc_df = pd.read_csv(cc_filepath)
    
    # plot
    cc_col = [c for c in cc_df.columns if c.startswith('channel')]
    bins = 15
    cc = np.zeros((len(cc_col), bins))
    for index, col in enumerate(cc_col):
        density, edge = np.histogram(cc_df[col], density=True, range=(-1, 1),
                                     bins=bins)
        cc[index, :] = density
        
    # get bin centers
    width = edge[1] - edge[0]
    center = edge[0:-1] + width/2
    
    # sort
    sk = np.argsort(cc_df[cc_col].mean(axis=0))[::-1]
    channelid = np.array([int(c.split('_')[1]) for c in cc_col])
    channelid, cc = channelid[sk], cc[sk, :]
    
    # plot first 3 for highlight
    color_list = ['tab:blue', 'tab:orange', 'tab:green']
    for index in range(3):
        plt.plot(center, cc[index, :], color=color_list[index])
        
    # plot the rest
    for index in range(3, cc.shape[0]):
        plt.plot(center, cc[index, :], 'k-', alpha=0.1)
        
    # custom legend
    marker_list = pd.read_csv(marker_filepath, header=None)[0].tolist()
    legend_elements = []
    for index in range(3):
        ch = channelid[index]
        ch_name = marker_list[ch]
        legend_elements.append(
            mlines.Line2D([0], [0], color=color_list[index], label=ch_name))
    legend_elements.append(
        mlines.Line2D([0], [0], color='black', alpha=0.1, label='others'))
    plt.legend(handles=legend_elements)
    
    plt.xlabel('correlation coefficient')
    plt.xticks(np.linspace(-1, 1, num=5))
    plt.ylabel('normalized histogram count')
    plt.yticks([])
    plt.title('PD1 (cytoplasm) vs. 22 markers (environment)')
    plt.savefig(output_fig_filepath)
    plt.show()