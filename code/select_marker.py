import os

import numpy as np
import pandas as pd

import polartk

if __name__ == '__main__':
    # path
    marker_filepath = os.path.expanduser('~/polar_data/data/markers.csv')
    marker_output_filepath = './selected_marker.csv'

    # get marker names
    with open(marker_filepath, 'r') as infile:
        marker_list = [line.strip() for line in infile.readlines()]
    old_index = list(range(len(marker_list)))
    dna_index = old_index[4::4] # keep DNA1
    background_index = [1, 2, 3]
    kept_index = list(set(old_index).difference(set(dna_index + background_index)))

    marker_df = pd.DataFrame(columns=['original_index', 'marker_name'],
            index=range(len(kept_index)))
    marker_df['original_index'] = kept_index
    marker_df['marker_name'] = marker_df['original_index'].apply(
            lambda x: marker_list[x])
    marker_df.reset_index(inplace=True)
    marker_df.rename(columns={'index': 'new_index'}, inplace=True)
    marker_df.to_csv(marker_output_filepath, index=False)
