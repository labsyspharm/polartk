import os

import numpy as np
import matplotlib.pyplot as plt

from skimage import io, measure, exposure, morphology
from skimage.external import tifffile

import polartk

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
    image_filepath = os.path.expanduser('~/polar_data/data/image.ome.tif')
    nuclei_mask_filepath = os.path.expanduser('~/polar_data/data/nuclei_mask.tif')
    cell_mask_filepath = os.path.expanduser('~/polar_data/data/cell_mask.tif')

    # params
    tile_shape = (15, 15)
    cellid = 1497

    # load data
    cell_mask = io.imread(cell_mask_filepath)
    nuclei_mask = io.imread(nuclei_mask_filepath)
    cell_region = [region for region in measure.regionprops(nuclei_mask)\
            if region.label == cellid][0]

    # open image
    channelid = 19 # PD1
    with tifffile.TiffFile(image_filepath) as tif:
        image = tif.series[0].pages[channelid].asarray()
        image = prep_intensity(image)

    change_list = [
            lambda arr: morphology.binary_erosion(arr, selem=morphology.disk(1)),
            lambda arr: arr,
            lambda arr: morphology.binary_dilation(arr, selem=morphology.disk(1)),
            ]
    result_list = []

    for change in change_list:
        # get cell coordinate from current nuclei mask
        bxl, byl, bxu, byu = cell_region.bbox
        bx, by = np.meshgrid(range(bxl, bxu), range(byl, byu), indexing='ij')
        bxc, byc = np.mean(bx[cell_region.image]), np.mean(by[cell_region.image])

        # calculate tile coordinates, based on centroid and tile shpae
        txl = int(np.round(bxc-tile_shape[0]/2)) # tile x lower bound
        tyl = int(np.round(byc-tile_shape[1]/2)) # tile y lower bound
        txu, tyu = txl+tile_shape[0], tyl+tile_shape[1] # tile x/y upper bound
        tcc = (bxc-txl, byc-tyl) # tile cell centroid
        image_xy = image[txl:txu, tyl:tyu]

        # adjust masks to assess sensitivity
        # ex. erosion and dilation
        cm = cell_mask[txl:txu, tyl:tyu] == cellid
        nm = nuclei_mask[txl:txu, tyl:tyu] == cellid
        nm = change(nm)

        # construct label based on segmentation masks
        label_xy = np.zeros_like(cm, dtype=int)
        label_xy[cm] = 1
        label_xy[nm] = 2

        # transform
        r_grid, t_grid, image_rt, label_rt = polartk.xy2rt(
                image=image_xy, centroid=tcc, label=label_xy)

        # calculate polarity
        mask = label_rt == 1
        t_profile = np.zeros_like(image_rt)
        t_profile[mask] = image_rt[mask]
        t_profile = t_profile.sum(axis=0) # sum over radius
        t_profile = t_profile[mask.any(axis=0)] # remove zeros from no data
        t_profile /= t_profile.sum() # normalization
        t = np.degrees(t_grid[0, mask.any(axis=0)]) # corresponding angles (in degrees)

        # append result for later plotting
        result = (tcc, label_xy, label_rt, image_xy, image_rt, t, t_profile)
        result_list.append(result)

    # plot
    fig, axes = plt.subplots(ncols=3, nrows=3, sharex=True, sharey=True,
            figsize=(9, 9))
    for col_id in range(axes.shape[1]):
        tcc, label_xy, label_rt, _, image_rt, _, _ = result_list[col_id]
        axes[0, col_id].imshow(label_xy, cmap='tab10')
        axes[0, col_id].scatter([tcc[1]], [tcc[0]], color='r', marker='o')
        axes[1, col_id].imshow(label_rt, cmap='tab10')
        axes[2, col_id].imshow(image_rt, cmap='gray')

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_title('shrink nuclei')
    axes[0, 1].set_title('original nuclei')
    axes[0, 2].set_title('expand nuclei')
    
    axes[0, 0].set_ylabel('label (XY)')
    axes[1, 0].set_ylabel(r'label (R$\theta$)')
    axes[2, 0].set_ylabel(r'image (R$\theta$)')

    fig.tight_layout()
    plt.savefig('../figures/demo_sensitivity_part1.png')
    plt.close()

    # plot image_xy and angular distribution
    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(4, 8))
    axes[0].imshow(result_list[1][3], cmap='gray')
    axes[0].set_title('original image (XY)')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].plot(result_list[0][-2], result_list[0][-1], color='k', linestyle='dotted',
            label='shrink ({:.3f})'.format(polartk.polarity(result_list[0][-1])))
    axes[1].plot(result_list[1][-2], result_list[1][-1], color='k', linestyle='dashed',
            label='original ({:.3f})'.format(polartk.polarity(result_list[1][-1])))
    axes[1].plot(result_list[2][-2], result_list[2][-1], color='k', linestyle='solid',
            label='expand ({:.3f})'.format(polartk.polarity(result_list[2][-1])))
    axes[1].legend()
    axes[1].set_title('Polarity')
    axes[1].set_xlabel(r'Angle ($\theta$, in degree)')
    axes[1].set_ylabel('normalized intensity')
    axes[1].set_yticks([])

    fig.tight_layout()
    plt.savefig('../figures/demo_sensitivity_part2.png')
    plt.close()
