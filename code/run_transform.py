import os
import csv
import shutil
import multiprocessing as mp

import numpy as np
import tqdm

from skimage import io, measure, exposure
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

def generate_job(image_filepath, channelid, nuclei_mask_filepath, cell_mask_filepath, tile_shape,
                 cell_selection_criteria=None, verbose=False):
    # load data
    cell_mask = io.imread(cell_mask_filepath)
    nuclei_mask = io.imread(nuclei_mask_filepath)

    # select cell centroids based on criteria
    if cell_selection_criteria is None:
        cc_dict = {region.label: region.centroid for region in measure.regionprops(nuclei_mask)}
    else:
        cc_dict = {region.label: region.centroid for region in measure.regionprops(nuclei_mask)\
                  if cell_selection_criteria(region)}

    # open image
    with tifffile.TiffFile(image_filepath) as tif:
        channel_image = tif.series[0].pages[channelid].asarray()
        channel_image = prep_intensity(channel_image)

        for cellid in tqdm.tqdm(cc_dict, desc='channel {}'.format(channelid), disable=not verbose):
            # calculate tile coordinates, based on centroid and tile shpae
            cc = cc_dict[cellid] # cell centroid
            txl = int(np.round(cc[0]-tile_shape[0]/2)) # tile x lower bound
            tyl = int(np.round(cc[1]-tile_shape[1]/2)) # tile y lower bound
            txu, tyu = txl+tile_shape[0], tyl+tile_shape[1] # tile x/y upper bound
            tcc = (cc[0]-txl, cc[1]-tyl) # tile cell centroid
            intensity = channel_image[txl:txu, tyl:tyu]

            # skip cells too close to borders, given tile shape
            valid_list = [
                    txl >= 0,
                    tyl >= 0,
                    txu < channel_image.shape[0],
                    tyu < channel_image.shape[1],
                    ]
            if not all(valid_list):
                continue

            # construct label based on segmentation masks
            cm = cell_mask[txl:txu, tyl:tyu] == cellid
            nm = nuclei_mask[txl:txu, tyl:tyu] == cellid
            label_xy = np.zeros_like(cm, dtype=int)
            label_xy[cm] = 1
            label_xy[nm] = 2

            # yield job
            job_args = dict(image=intensity, centroid=tcc, label=label_xy)

            yield cellid, job_args

def process_job(job):
    cellid, job_args = job
    
    r_grid, t_grid, image_rt, label_rt = polartk.xy2rt(**job_args)

    num_row = r_grid.shape[0]*r_grid.shape[1]
    num_col = 5 # N, R, Theta, M (mask label), Intensity at current channel
    array = np.zeros((num_row, num_col))
    array[:, 0] = cellid
    array[:, 1] = r_grid.flatten()
    array[:, 2] = t_grid.flatten()
    array[:, 3] = label_rt.flatten()
    array[:, 4] = image_rt.flatten()
    return array

if __name__ == '__main__':
    # params
    tile_shape = (15, 15)
    image_filepath = '~/polar_data/data/image.ome.tif'
    nuclei_mask_filepath = '~/polar_data/data/nuclei_mask.tif'
    cell_mask_filepath = '~/polar_data/data/cell_mask.tif'
    output_header = ['cellid', 'r', 'theta', 'label', 'intensity']
    output_folderpath = '~/polar_data/transformed_result'
    
    # prepare output folder
    if os.path.isdir(output_folderpath):
        shutil.rmtree(output_folderpath)
    os.mkdir(output_folderpath)
    
    # cell selection criteria
    def cell_criteria(region):
        checklist = [
                region.solidity > 0.9,
                region.area > 30,
                region.area < 200,
                ]
        return all(checklist)

    # define channel list to process
    with tifffile.TiffFile(image_filepath) as tif:
        num_channel = len(tif.series[0].pages) -2 # last two channels are masks for this ome.tif
    channel_list = list(range(num_channel))
    dna_list = channel_list[4::4] # keep DNA1
    background_list = [1, 2, 3]
    channel_list = list(set(channel_list).difference(set(dna_list + background_list)))
    
    # parallel processing
    wp = mp.Pool(os.cpu_count()) # worker pool
   
    for channelid in channel_list:
        output_filepath = os.path.join(output_folderpath, 'channel_{}.csv'.format(channelid))
        with open(output_filepath, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(output_header)
        
            for array in wp.imap_unordered(func=process_job, iterable=generate_job(
                image_filepath=image_filepath, channelid=channelid,
                nuclei_mask_filepath=nuclei_mask_filepath,
                cell_mask_filepath=cell_mask_filepath, tile_shape=tile_shape,
                cell_selection_criteria=cell_criteria, verbose=True)):
            
                csvwriter.writerows(array)

    print('all done.')
