import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import interpolate
from skimage.external import tifffile
from skimage import measure, io, exposure

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
    image_filepath = os.path.join(data_folderpath, 'data', 'image.ome.tif')
    nuclei_mask_filepath = os.path.join(data_folderpath, 'data', 'nuclei_mask.tif')

    talign_filepath = os.path.join(data_folderpath, 'pd1_angular_alignment.csv')

    # params
    view_shape = (100, 100)

    # load data
    with tifffile.TiffFile(image_filepath) as infile:
        dna = infile.series[0].pages[0].asarray()
        sox10 = infile.series[0].pages[7].asarray()
        pd1 = infile.series[0].pages[19].asarray()

    dna = prep_intensity(dna)
    sox10 = prep_intensity(sox10)
    pd1 = prep_intensity(pd1)

    nuclei_mask = io.imread(nuclei_mask_filepath)
    talign_df = pd.read_csv(talign_filepath)

    region_dict = {r.label:r for r in measure.regionprops(nuclei_mask)}
    talign_df['xc'] = talign_df['cellid'].apply(
            lambda x: region_dict[x].centroid[0])
    talign_df['yc'] = talign_df['cellid'].apply(
            lambda x: region_dict[x].centroid[1])

    rgb_image = np.stack([sox10, pd1, np.zeros_like(dna)], axis=-1)
    for c in range(3):
        rgb_image[..., c] = np.maximum(rgb_image[..., c], dna*0.5)

    talign_df.sort_values('polarity', ascending=False, inplace=True)
    talign_df.reset_index(drop=True, inplace=True)

    for index, row in talign_df.iterrows():
        # construct vector
        r = 5 * row['polarity']
        t = row['angle']
        dx = r * np.cos(np.radians(t))
        dy = r * np.sin(np.radians(t))
        # select image
        region = region_dict[row['cellid']]
        xc, yc = region.centroid
        xl = int(np.round(xc-view_shape[0]/2))
        yl = int(np.round(yc-view_shape[1]/2))
        xu, yu = xl+view_shape[0], yl+view_shape[1]
        # check border
        valid_list = [xl >= 0, yl >= 0, xu < dna.shape[1], yu < dna.shape[0]]
        if not all(valid_list):
            continue
        plt.imshow(rgb_image[xl:xu, yl:yu])
        plt.arrow(yc-yl-dx, xc-xl-dy, dx, dy,
                width=0.3, head_width=3, color='magenta')
        plt.xticks([])
        plt.yticks([])
        plt.title('cell ID: {:.0f}'.format(row['cellid']))
        plt.show()
        plt.close()
