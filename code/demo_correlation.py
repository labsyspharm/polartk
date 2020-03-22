import os

import numpy as np
import pandas as pd

if __name__ == '__main__':
    # paths
    data_folderpath = os.path.expanduser('~/polar_data')
#    target_filepath = os.path.join(data_folderpath, 'pd1_angular_alignment.csv')
    tdist_filepath = os.path.join(data_folderpath, 'angular_distribution_30x30.csv')
    output_filepath = os.path.join(data_folderpath, 'pd1_env_corrcoef.csv')
    
    # load data
#    target_df = pd.read_csv(target_filepath)
#    target_df = target_df[['cellid']]
    tdist_df = pd.read_csv(tdist_filepath)
    has_cyto = tdist_df.loc[tdist_df['label'] == 'cytoplasm', ['cellid']]
    has_env = tdist_df.loc[tdist_df['label'] == 'environment', ['cellid']]
    target_df = has_cyto.merge(has_env, on='cellid', how='inner')
    tdist_df = tdist_df.merge(target_df, on='cellid', how='inner')

    # pd1 in cytoplasm as one arm
    pd1_col = [c for c in tdist_df.columns if '_ch19_' in c]
    pd1_mask = tdist_df['label'] == 'cytoplasm'
    pd1_df = tdist_df.loc[pd1_mask]
    pd1_dict = {}
    for index, row in pd1_df.iterrows():
        pd1_dict[row['cellid']] = row[pd1_col].values.astype(float)
    
    # other 22 markers in environment as the other arm
    env_mask = tdist_df['label'] == 'environment'
    int_col = [c for c in tdist_df.columns if c.startswith('intensity')]
    env_cellid = tdist_df.loc[env_mask, 'cellid'].tolist()

    mapping = {c:int(c.split('_')[1][2:]) for c in int_col}
    gb = tdist_df.loc[env_mask, int_col]\
        .groupby(mapping, axis=1)
    df_list = []
    for gkey in gb.groups:
        g = gb.get_group(gkey).copy()
        sep = '_ch{:.0f}_'.format(gkey)
        g.columns = [c.split(sep)[1] for c in g.columns]
        g['cellid'] = env_cellid
        g['channel'] = gkey
        df_list.append(g)
        
    env_df = pd.concat(df_list, axis=0)

    # calculate correlation coefficient
    def corr_fn(row):
        cellid = row.values[-2]
        env_intensity = row.values[0:-2]
        pd1_intensity = pd1_dict[cellid]
        env_mask = np.isfinite(env_intensity)
        pd1_mask = np.isfinite(pd1_intensity)
        mask = env_mask & pd1_mask
        if mask.all():
            cc = np.corrcoef(pd1_intensity[mask], env_intensity[mask])
            return cc[0, 1]
        else:
            return float('nan')
    
    env_df['corrcoef'] = env_df.apply(corr_fn, axis=1)
    cc_df = env_df[['cellid', 'channel', 'corrcoef']].copy()
    cc_df['channel'] = cc_df['channel'].apply(lambda x: 'channel_{:.0f}'.format(x))
    cc_df = cc_df.pivot(index='cellid', columns='channel', values='corrcoef')

    # save file
    cc_df.to_csv(output_filepath)
