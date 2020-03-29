import os
import functools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import io, measure, exposure, segmentation
from skimage.external import tifffile

if __name__ == '__main__':
    # paths
    data_folderpath = os.path.expanduser('~/polar_data')
    input_filepath = './pd1_polarity.csv'
    image_filepath = os.path.join(data_folderpath, 'data', 'image.ome.tif')
    cell_mask_filepath = os.path.join(data_folderpath, 'data', 'cell_mask.tif')
    marker_filepath = './selected_marker.csv'
    output_filepath = os.path.expanduser('~/polartk/figures/fig3b.png')

    # params
    visualize_scope = (100, 100)

    # load data
    df = pd.read_csv(input_filepath)
    criteria = [
            df['polarity_PD1'] > 0.2,
            df['intensity_SOX10'].apply(np.log10) < 5.7,
            df['intensity_PD1'].apply(np.log10) > 6,
            ]
    mask = functools.reduce(lambda x, y: x & y, criteria)
    df = df.loc[mask]

    marker_df = pd.read_csv(marker_filepath)

    cell_mask = io.imread(cell_mask_filepath)
    cell_mask = cell_mask.astype(int)
    cell_region_dict = {region.label: region for region in\
            measure.regionprops(label_image=cell_mask)}

    # load image
    image_list = []
    target_list = ['SOX10', 'PD1', 'DNA1']
    with tifffile.TiffFile(image_filepath) as tif:
        for target in target_list:
            index = marker_df.loc[marker_df['marker_name']==target,
                    'original_index'].tolist()[0]
            image = tif.series[0].pages[index].asarray()
            image = image.astype(float)
            image = exposure.rescale_intensity(image, out_range=(0, 1),
                    in_range=tuple(np.percentile(image, (1, 99))))
            image_list.append(image)
    image = np.stack(image_list, axis=-1) # RGB

    # plot
    df.sort_values('polarity_PD1', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    for index in df.index:
        # calculate scope
        xc, yc = df.loc[index, ['centroid-0', 'centroid-1']]
        txl = int(np.round(xc-visualize_scope[0]/2))
        tyl = int(np.round(yc-visualize_scope[1]/2))
        txu, tyu = txl+visualize_scope[0], tyl+visualize_scope[1]

        # make sure scope does not exceed image range
        checklist = [txl >= 0, tyl >= 0, txu < image.shape[0], tyu < image.shape[1]]
        if not all(checklist):
            continue

        # calculate cell outline
        cellid = df.loc[index, 'label']
        cell_region = cell_region_dict[cellid]
        m = np.pad(cell_region.image, pad_width=1, constant_values=0)
        cell_outline = segmentation.find_boundaries(m, mode='outer')
        outline_index = np.argwhere(cell_outline)
        outline_index -= 1 # offset the padding
        bxl, byl, _, _ = cell_region.bbox
        outline_index[:, 0] += (bxl - txl)
        outline_index[:, 1] += (byl - tyl)

        cell_image = image[txl:txu, tyl:tyu, :].copy()
        cell_image[outline_index[:, 0], outline_index[:, 1], :] = 1

        # plot
        print(df.loc[index])
        plt.imshow(cell_image)
        plt.title(cellid)
        plt.show()
        plt.close()
