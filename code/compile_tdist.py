import os
import functools
import glob

import numpy as np
import pandas as pd
import tqdm

if __name__ == '__main__':
    # paths
    data_filepattern = os.path.expanduser('~/polar_data/transformed_result/channel_*.csv')
    output_filepath = os.path.expanduser('~/polar_data/angular_distribution.csv')
    
    # params
    label_dict = {0: 'environment', 1: 'cytoplasm', 2: 'nucleus'}
    merge_fn = lambda x, y: pd.merge(x, y, on=['cellid', 'label'], how='outer')

    # process each file
    df_list = []
    for data_filepath in tqdm.tqdm(glob.glob(data_filepattern)):
        df = pd.read_csv(data_filepath)
        df = df.groupby(by=['cellid', 'theta', 'label'])['intensity'].sum()
        df = df.unstack(level=1) # pivot only theta, leaving cellid and label in the index
        df.sort_index(axis=1, ascending=True, inplace=True) # sort theta
        
        channelid = os.path.splitext(os.path.basename(data_filepath))[0].split('_')[1]
        df.columns = ['intensity_ch{}_theta{}'.format(channelid, i) for i in range(len(df.columns))]
        df = df.reset_index()
        
        df['cellid'] = df['cellid'].astype(int)
        df['label'] = df['label'].map(label_dict)
        
        df_list.append(df)

    # merge
    df = functools.reduce(merge_fn, df_list)
    df.to_csv(output_filepath, index=False)
