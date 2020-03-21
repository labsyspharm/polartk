import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # paths
    data_folderpath = os.path.expanduser('~/polar_data')
    cc_filepath = os.path.join(data_folderpath, 'pd1_env_corrcoef.csv')
    
    # load data
    cc_df = pd.read_csv(cc_filepath)
    
    # plot
    cc_col = [c for c in cc_df.columns if c.startswith('channel')]
    for col in cc_col:
        density, edge = np.histogram(cc_df[col], density=True,
                                    range=(-1, 1), bins=15)
        width = edge[1] - edge[0]
        center = edge[0:-1] + width/2
        plt.plot(center, density, 'k-', alpha=0.1)
        
    plt.xlabel('correlation coefficient')
    plt.xticks(np.linspace(-1, 1, num=5))
    plt.ylabel('normalized histogram count')
    plt.yticks([])
    plt.title('PD1 (cytoplasm) vs. 22 markers (environment)')
    plt.show()