import os
import functools

import numpy as np
import pandas as pd
import tqdm

from skimage import io, measure
from skimage.external import tifffile

import polartk

if __name__ == '__main__':
    # path
    data_folderpath = os.path.expanduser('~/polar_data/')
    transform_result_folderpath = os.path.join(data_folderpath,
            'transform_output_30x30')
    nuclei_mask_filepath = os.path.join(data_folderpath, 'data', 'nuclei_mask.tif')
    cell_mask_filepath = os.path.join(data_folderpath, 'data', 'cell_mask.tif')
    image_filepath = os.path.join(data_folderpath, 'data', 'image.ome.tif')

    marker_filepath = './selected_marker.csv'
    output_filepath = './pd1_polarity.csv'

    # get index
    marker_df = pd.read_csv(marker_filepath)
    index_pd1 = marker_df.loc[marker_df['marker_name']=='PD1', 'new_index']\
            .tolist()[0]

    # get polarity
    result_list = []
    for filename in tqdm.tqdm(os.listdir(transform_result_folderpath)):
        filepath = os.path.join(transform_result_folderpath, filename)
        array = np.load(filepath)
        label = array[..., 0]
        valid_mask = (label == 1).any(axis=0) # any cytoplasm at the angle
        pd1 = array[..., index_pd1]
        pd1 = pd1.sum(axis=0)[valid_mask]
        p = polartk.polarity(pd1)
        cellid = int(os.path.splitext(filename)[0][len('output_job_'):])
        result_list.append({'label': cellid, 'polarity_PD1': p})

    polarity_df = pd.DataFrame.from_records(result_list)

    # get intensities
    target_list = ['DNA1', 'PD1', 'SOX10']
    df_list = []
    cell_mask = io.imread(cell_mask_filepath)
    
    with tifffile.TiffFile(image_filepath) as tif:
        for target in target_list:
            target_index = marker_df.loc[marker_df['marker_name']==target,
                    'original_index'].tolist()[0]
            image = tif.series[0].pages[target_index].asarray(memmap=True)
            out_dict = measure.regionprops_table(label_image=cell_mask,
                    intensity_image=image, properties=['label', 'area',
                        'mean_intensity'])
            out_df = pd.DataFrame(out_dict)
            col_name = 'intensity_{}'.format(target)
            out_df[col_name] = out_df['area'] * out_df['mean_intensity']
            df_list.append(out_df[['label', col_name]])

    # get centroids
    nuclei_mask = io.imread(nuclei_mask_filepath)
    out_dict = measure.regionprops_table(label_image=nuclei_mask,
            properties=['label', 'centroid'])
    centroid_df = pd.DataFrame(out_dict)

    # merge
    merge_fn = lambda x, y: x.merge(y, on='label', how='inner')
    df = functools.reduce(merge_fn, df_list + [polarity_df, centroid_df])

    df.to_csv(output_filepath, index=False)
