import os
import functools
import glob

import numpy as np
import pandas as pd
import tqdm

import polartk

if __name__ == '__main__':
    # paths
    data_filepattern = os.path.expanduser(
            '~/polar_data/polarity_sensitivity_analysis/channel_19_*.csv')
    output_filepath = os.path.expanduser(
            '~/polar_data/polarity_sensitivity_analysis/summary.csv')
    
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
        
        case_name = os.path.splitext(os.path.basename(data_filepath))[0].split('_19_')[1]
        df = df.apply(polartk.polarity, axis=1)
        df = df.reset_index(name='polarity_{}'.format(case_name))
        df['cellid'] = df['cellid'].astype(int)
        df['label'] = df['label'].map(label_dict)
        
        df_list.append(df)

    # merge
    df = functools.reduce(merge_fn, df_list)
    df.to_csv(output_filepath, index=False)
