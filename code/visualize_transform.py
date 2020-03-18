import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from scipy import stats
from skimage import io, measure, segmentation, exposure
from skimage.external import tifffile

import polartk

def prep_intensity(arr):
    '''
    Stretch image contrast.
    '''
    arr = arr.astype(float)
    arr = exposure.rescale_intensity(arr, out_range=(0, 1),
            in_range=tuple(np.percentile(arr, (1, 99))))
    return arr

def cell_criteria(region):
    '''
    Cell selection criteria.
    '''
    checklist = [
            region.solidity > 0.9,
            region.area > 30,
            region.area < 200,
            ]
    return all(checklist)

if __name__ == '__main__':
    # paths
    nuclei_mask_filepath = '~/polar_data/data/nuclei_mask.tif'
    cell_mask_filepath = '~/polar_data/data/cell_mask.tif'
    image_filepath = '~/polar_data/data/image.ome.tif'
    
    # params
    tile_shape = (15, 15)
    
    # load data
    nuclei_mask = io.imread(nuclei_mask_filepath)
    cell_mask = io.imread(cell_mask_filepath)
    
    with tifffile.TiffFile(image_filepath) as tif:
        dna_intensity = tif.series[0].pages[0].asarray()
        cyto_intensity = tif.series[0].pages[19].asarray()
        cyto_name = 'PD1'
    
    dna_intensity = prep_intensity(dna_intensity)
    cyto_intensity = prep_intensity(cyto_intensity)

    # select and match regions
    region_dict = {}
    for nuclei_region in measure.regionprops(nuclei_mask):
        if cell_criteria(nuclei_region):
            region_dict[nuclei_region.label] = {'nuclei': nuclei_region}
            
    for cell_region in measure.regionprops(cell_mask):
        if cell_region.label in region_dict:
            region_dict[cell_region.label]['cell'] = cell_region

    # random sampling of cells
    all_label = list(region_dict.keys())
    np.random.shuffle(all_label)

    for i in all_label:
        # unpack
        cell_region = region_dict[i]['cell']
        nuclei_region = region_dict[i]['nuclei']

        # unify coordinate
        cc = nuclei_region.centroid
        cxl, cyl, cxu, cyu = cell_region.bbox
        txl = int(np.round(cc[0]-tile_shape[0]/2))
        tyl = int(np.round(cc[1]-tile_shape[1]/2))
        txu, tyu = txl+tile_shape[0], tyl+tile_shape[1]
        xl, xu = min(cxl, txl), max(cxu, txu)
        yl, yu = min(cyl, tyl), max(cyu, tyu)
        cc = (cc[0]-xl, cc[1]-yl)
        
        # make label
        cm = cell_mask[xl:xu, yl:yu] == i
        nm = nuclei_mask[xl:xu, yl:yu] == i
        label_xy = np.zeros_like(cm, dtype=int)
        label_xy[cm] = 1
        label_xy[nm] = 2
        
        # slice markers
        dna_xy = dna_intensity[xl:xu, yl:yu]
        cyto_xy = cyto_intensity[xl:xu, yl:yu]
    
        # transform
        r_grid, t_grid, dna_rt, label_rt = polartk.xy2rt(image=dna_xy, centroid=cc, label=label_xy)
        _, _, cyto_rt, _ = polartk.xy2rt(image=cyto_xy, centroid=cc, label=label_xy)

        # draw tile in plot coordinate, handle boundary issues
        tile_patch = mpatches.Rectangle(xy=(tyl-yl, txl-xl),
                width=tile_shape[1]-1, height=tile_shape[0]-1,
                facecolor='none', edgecolor='r')

        # plot
        fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12,8), sharex=True, sharey=True)

        axes[0, 0].imshow(label_xy, cmap='tab10')
        axes[0, 0].add_patch(tile_patch)
        axes[0, 0].scatter([cc[1]], [cc[0]], c='r', marker='o')
        axes[0, 1].imshow(dna_xy, cmap='gray')
        axes[0, 2].imshow(cyto_xy, cmap='gray')
        axes[1, 0].imshow(label_rt, cmap='tab10')
        axes[1, 1].imshow(dna_rt, cmap='gray')
        axes[1, 2].imshow(cyto_rt, cmap='gray')

        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        axes[1, 0].set_xlabel(r'$\theta$')
        axes[1, 0].set_ylabel('R')

        axes[0, 0].set_title('masks')
        axes[0, 1].set_title('DNA1')
        axes[0, 2].set_title(cyto_name)

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        # custom legend
        legend_elements = [
                mlines.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor='r', markeredgecolor='r',
                    label='cell centroid'),
                mpatches.Patch(facecolor='none', edgecolor='r',
                    label='tile (15x15 pixels)'),
                mpatches.Patch(facecolor='tab:cyan', edgecolor='tab:cyan',
                    label='nuclei'),
                mpatches.Patch(facecolor='tab:brown', edgecolor='tab:brown',
                    label='cytoplasm'),
                mpatches.Patch(facecolor='tab:blue', edgecolor='tab:blue',
                    label='environment'),
                ]
        plt.figlegend(handles=legend_elements, ncol=5, bbox_to_anchor=(0.95, 0.99),
                loc='upper right')

        fig.suptitle('cell ID: {}'.format(i), y=0.98, x=0.15, ha='left')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
