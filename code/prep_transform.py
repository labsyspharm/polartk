import os
import shutil

import numpy as np
import tqdm

from skimage import io, measure
from skimage.external import tifffile

if __name__ == '__main__':
    # paths and params
    tile_shape = (30, 30)
    data_folderpath = os.path.expanduser('~/polar_data')
    image_filepath = os.path.join(data_folderpath, 'data', 'image.ome.tif')
    nuclei_mask_filepath = os.path.join(data_folderpath, 'data', 'nuclei_mask.tif')
    cell_mask_filepath = os.path.join(data_folderpath, 'data', 'cell_mask.tif')
    output_folderpath = os.path.join(data_folderpath, 'transform_job_30x30')
    
    # prepare output folder
    if os.path.isdir(output_folderpath):
        shutil.rmtree(output_folderpath)
    os.mkdir(output_folderpath)

    # load data
    nuclei_mask = io.imread(nuclei_mask_filepath)
    cell_mask = io.imread(cell_mask_filepath)
    region_dict = {}
    for region in measure.regionprops(nuclei_mask):
        checklist = [
                region.solidity > 0.9,
                region.area > 30,
                region.area < 200,
                ]
        if all(checklist):
            region_dict[region.label] = {'nuclei': region}
    for region in measure.regionprops(cell_mask):
        if region.label in region_dict:
            region_dict[region.label]['cell'] = region
    for cellid in region_dict:
        if len(region_dict[cellid]) < 2:
            del region_dict[cellid]
    
    # main loop
    with tifffile.TiffFile(image_filepath) as tif:
        # select channels
        num_channel = len(tif.series[0].pages)-2 # last two are masks for this file
        channel_list = list(range(num_channel))
        dna_list = channel_list[4::4] # keep DNA1
        background_list = [1, 2, 3]
        channel_list = list(set(channel_list).difference(set(dna_list + background_list)))

        # pre-calculate percentiles for clipping
        pct_dict = {}
        for index_channel in channel_list:
            image = tif.series[0].pages[index_channel].asarray(memmap=True)
            image = image.astype(float)
            pct_dict[index_channel] = np.percentile(image, (1, 99))

        image_shape = tif.series[0].pages[0].shape
        for cellid in tqdm.tqdm(region_dict):
            # define tile region
            nuclei_region = region_dict[cellid]['nuclei']
            cell_region = region_dict[cellid]['cell']
            tcc = nuclei_region.centroid
            txl = int(np.round(tcc[0]-tile_shape[0]/2))
            tyl = int(np.round(tcc[1]-tile_shape[1]/2))
            txu, tyu = txl+tile_shape[0], tyl+tile_shape[1]

            # skip cells too close to borders, given tile shape
            checklist = [
                    txl >= 0,
                    tyl >= 0,
                    txu < image_shape[0],
                    tyu < image_shape[1],
                    ]
            if not all(checklist):
                continue

            # make label
            nm = nuclei_mask[txl:txu, tyl:tyu] == cellid
            cm = cell_mask[txl:txu, tyl:tyu] == cellid
            label = np.zeros(tile_shape)
            label[cm] = 1
            label[nm] = 2

            # concat job data
            job_array = np.zeros(tile_shape+(1+len(channel_list),))
            job_array[..., 0] = label
            for i, index_channel in enumerate(channel_list):
                # load image
                image = tif.series[0].pages[index_channel].asarray(memmap=True)
                # slice
                image = image[txl:txu, tyl:tyu].astype(float)
                # rescale intensity to (0, 1)
                pct_low, pct_high = pct_dict[index_channel]
                image = np.clip(image, a_min=pct_low, a_max=pct_high)
                image -= pct_low
                image /= (pct_high-pct_low)
                job_array[..., i+1] = image

            # save to disk
            output_filepath = os.path.join(output_folderpath,
                    'job_{}.npy'.format(cellid))
            np.save(output_filepath, job_array)

    print('all done.')
