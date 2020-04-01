import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.external import tifffile
from skimage import io, exposure, measure

def prep_intensity(arr):
    '''
    function to stretch contrast
    '''
    arr = arr.astype(float)
    arr = exposure.rescale_intensity(arr, out_range=(0, 1),
            in_range=tuple(np.percentile(arr, (1, 99))))
    return arr

if __name__ == '__main__':
    # paths
    data_folderpath = os.path.expanduser('~/polar_data')
    cc_filepath = os.path.join(data_folderpath, 'pd1_env_corrcoef.csv')
    marker_filepath = os.path.join(data_folderpath, 'data', 'markers.csv')
    
    image_filepath = os.path.join(data_folderpath, 'data', 'image.ome.tif')
    nuclei_mask_filepath = os.path.join(data_folderpath, 'data', 'nuclei_mask.tif')
    cell_mask_filepath = os.path.join(data_folderpath, 'data', 'cell_mask.tif')
    
    # params
    tile_shape = (30, 30)
    
    # load data
    cc_df = pd.read_csv(cc_filepath)
    marker_list = pd.read_csv(marker_filepath, header=None)[0].tolist()
    
    name_list = ['DNA1', 'PD1', 'CD68', 'MITF']
    image_dict = {}
    
    nuclei_mask = io.imread(nuclei_mask_filepath)
    cell_mask = io.imread(cell_mask_filepath)
    nuclei_region_dict = {region.label:region for region in\
                     measure.regionprops(nuclei_mask)}
    cell_region_dict = {region.label:region for region in\
                     measure.regionprops(cell_mask)}
    
    with tifffile.TiffFile(image_filepath) as tif:
        for name in name_list:
            index_name = marker_list.index(name)
            im = tif.series[0].pages[index_name].asarray()
            image_dict[name] = prep_intensity(im)
        
    # test
    index_cd68 = marker_list.index('CD68')
    cc_df.sort_values('channel_{}'.format(index_cd68), ascending=False,
                      inplace=True)
    cc_df.reset_index(drop=True, inplace=True)

    for index in cc_df.index:
        print(cc_df.loc[index])
        cellid, corrcoef = cc_df.loc[index,
                ['cellid', 'channel_{}'.format(index_cd68)]]
        # calculate tile coords
        cell_centroid = nuclei_region_dict[cellid].centroid
        txl = int(np.round(cell_centroid[0]-tile_shape[0]/2))
        tyl = int(np.round(cell_centroid[1]-tile_shape[1]/2))
        txu, tyu = txl+tile_shape[0], tyl+tile_shape[1]
        
        # compile image
        rgb_image = np.stack([
            image_dict['CD68'][txl:txu, tyl:tyu],
            image_dict['PD1'][txl:txu, tyl:tyu],
            image_dict['DNA1'][txl:txu, tyl:tyu],
        ], axis=-1)
        
        # make cell label
        nm = nuclei_mask[txl:txu, tyl:tyu]
        cm = cell_mask[txl:txu, tyl:tyu]
        cell_label = np.zeros_like(nm)
        cell_label[cm == cellid] = 1
        cell_label[nm == cellid] = 2
        
        # plot
        fig, axes = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True,
                                 figsize=(6, 3))
        axes[0].imshow(rgb_image)
        axes[0].set_title('RGB=(CD68, PD1, DNA1)')
        axes[1].imshow(cell_label, cmap='tab10')
        axes[1].set_title('cell mask label')
        
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            
        fig.suptitle('cell ID: {}, corrcoef: {:.2e}'.format(cellid, corrcoef))
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        plt.show()
