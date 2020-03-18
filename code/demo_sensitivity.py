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

    # load data
    cell_mask = io.imread(cell_mask_filepath)
    nuclei_mask = io.imread(nuclei_mask_filepath)

    # select cell centroids based on criteria
    cellid = 1497
    cc = [region.centroid for region in measure.regionprops(nuclei_mask)\
            if region.label == cellid][0] # cell centroid

    # open image
    channelid = 19 # PD1
    with tifffile.TiffFile(image_filepath) as tif:
        image = tif.series[0].pages[channelid].asarray()
        image = prep_intensity(image)

    # calculate tile coordinates, based on centroid and tile shpae
    txl = int(np.round(cc[0]-tile_shape[0]/2)) # tile x lower bound
    tyl = int(np.round(cc[1]-tile_shape[1]/2)) # tile y lower bound
    txu, tyu = txl+tile_shape[0], tyl+tile_shape[1] # tile x/y upper bound
    tcc = (cc[0]-txl, cc[1]-tyl) # tile cell centroid
    intensity = image[txl:txu, tyl:tyu]

    # adjust masks to assess sensitivity
    # ex. erosion and dilation
    cm = cell_mask[txl:txu, tyl:tyu] == cellid
    nm = nuclei_mask[txl:txu, tyl:tyu] == cellid
    nm = morphology.binary_dilation(nm, selem=morphology.disk(1))

    # construct label based on segmentation masks
    label_xy = np.zeros_like(cm, dtype=int)
    label_xy[cm] = 1
    label_xy[nm] = 2

    # transform
    r_grid, t_grid, image_rt, label_rt = polartk.xy2rt(
            image=intensity, centroid=tcc, label=label_xy)

    # calculate polarity
    mask = label_rt == 1
    t_profile = np.zeros_like(image_rt)
    t_profile[mask] = image_rt[mask]
    t_profile = t_profile.sum(axis=0) # sum over radius
    t_profile = t_profile[mask.any(axis=0)] # remove zeros from no data
    t_profile /= t_profile.sum() # normalization
    t = np.degrees(t_grid[0, mask.any(axis=0)]) # corresponding angles (in degrees)

    plt.plot(t, t_profile)
    plt.title('polarity: {:.3f}'.format(polartk.polarity(t_profile)))
    plt.ylim([0, 1])
    plt.show()

