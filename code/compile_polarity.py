import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

import polartk

if __name__ == '__main__':
    # paths
    input_filepath = os.path.expanduser('~/polar_data/angular_distribution.csv')
    output_filepath = os.path.expanduser('~/polar_data/polarity.csv')

    # load data
    df = pd.read_csv(input_filepath)
    channel_col = [c for c in df.columns if '_ch' in c]

    # column grouping
    get_channel = lambda s: int(s.split('_ch')[1].split('_')[0])
    col_dict = {c: get_channel(c) for c in channel_col}
    unique_channel = set(list(col_dict.values()))
    ch_gb = df[channel_col].groupby(col_dict, axis=1)

    # prepare containers
    p_col = ['polarity_ch{}'.format(c) for c in unique_channel]
    result_df = pd.DataFrame(columns=p_col, index=df.index)
    result_df[['cellid', 'label']] = df[['cellid', 'label']]

    for ch_gkey in tqdm.tqdm(ch_gb.groups, total=len(unique_channel)):
        ch_df = ch_gb.get_group(ch_gkey)
        ch_df = ch_df.sort_index(axis=1)
        result_df['polarity_ch{}'.format(ch_gkey)] = ch_df.apply(polartk.polarity, axis=1)

    result_df.to_csv(output_filepath, index=False)

