import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import interpolate

if __name__ == '__main__':
    # paths
    data_folderpath = os.path.expanduser('~/polar_data')

    tdist_filepath = os.path.join(data_folderpath, 'angular_distribution.csv')
    sensitivity_filepath = os.path.join(data_folderpath,
            'polarity_sensitivity_analysis', 'summary.csv')

    output_fig_filepath = '../figures/pd1_angular_alignment.png'
    output_csv_filepath = os.path.join(data_folderpath, 'pd1_angular_alignment.csv')

    # params
    sensitivity_threshold = 0.13 # from random sampling

    # filter cell
    sensitivity_df = pd.read_csv(sensitivity_filepath)
    pcol = [c for c in sensitivity_df.columns if c.startswith('polarity')]
    valid_mask = (sensitivity_df['label'] == 'cytoplasm')\
            & (sensitivity_df[pcol].min(axis=1) > sensitivity_threshold)
    sensitivity_df = sensitivity_df.loc[valid_mask]
    valid_cellid = set(sensitivity_df['cellid'].tolist())

    # get angular distr
    tdist_df = pd.read_csv(tdist_filepath)
    valid_mask = (tdist_df['label'] == 'cytoplasm')\
            & tdist_df['cellid'].apply(lambda x: x in valid_cellid)
    tcol = [c for c in tdist_df.columns if c.startswith('intensity_ch19')]
    tdist_df = tdist_df.loc[valid_mask, ['cellid'] + tcol]
    tdist_df.sort_index(axis=1, inplace=True)

    # allow up to one missing data
    valid_mask = tdist_df.isnull().sum(axis=1) <= 1
    tdist_df = tdist_df.loc[valid_mask]

    # find angle
    t = np.linspace(0, 2*np.pi, num=15, endpoint=False)
    t_grid = np.linspace(0, 2*np.pi, num=100, endpoint=False)
    t, t_grid = np.degrees(t), np.degrees(t_grid)
    angle = np.zeros(tdist_df.shape[0])
    tdist_df.reset_index(inplace=True, drop=True)

    # also collect for plotting
    index_anchor = 50
    y_pred_all = np.zeros((tdist_df.shape[0], t_grid.shape[0]))

    for index, row in tdist_df.iterrows():
        # handle missing data
        y = row[tcol].values.flatten()
        mask = np.isfinite(y)
        x, y = t[mask], y[mask]
        # create padding for border issues
        xp = np.hstack([x-360, x, x+360])
        yp = np.tile(y, 3)
        interp_fn = interpolate.interp1d(xp, yp, kind='cubic')
        y_pred = interp_fn(t_grid)
        # shift to align max
        angle[index] = t_grid[np.argmax(y_pred)]
        # align for plotting
        y_pred = np.roll(y_pred, index_anchor - np.argmax(y_pred))
        # normalize
        y_pred /= y_pred.sum()

        y_pred_all[index, :] = y_pred

    # plot traces
    for i in range(y_pred_all.shape[0]):
        plt.plot(t_grid, y_pred_all[i, :], 'k-', alpha=0.02)

    plt.plot(t_grid, y_pred_all.mean(axis=0), 'k-')
    plt.savefig(output_fig_filepath)
    plt.close()

    # make dataframe
    df = sensitivity_df[['cellid', 'polarity_nxcx']].copy()
    df = df.merge(tdist_df[['cellid']], on='cellid', how='right')
    df.columns = ['cellid', 'polarity']
    df['angle'] = angle
    df.to_csv(output_csv_filepath, index=False)
