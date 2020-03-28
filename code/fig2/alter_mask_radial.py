import os

import numpy as np

from skimage import morphology

if __name__ == '__main__':
    # paths
    marker_filepath = os.path.expanduser('~/polar_data/data/markers.csv')
    xydata_filepath = 'job_1497.npy'

    # get marker list
    with open(marker_filepath, 'r') as infile:
        marker_list = [line.strip() for line in infile.readlines()]
    all_index = list(range(len(marker_list)))
    dna_index = list(range(4, len(marker_list), 4))
    background_index = [1, 2, 3]
    marker_index = set(all_index).difference(set(dna_index))\
            .difference(set(background_index))
    marker_index = list(marker_index)
    index_pd1 = marker_index.index(marker_list.index('PD1'))

    # load data
    xydata = np.load(xydata_filepath)
    label_xy = xydata[..., 0]
    nuclei_mask = label_xy == 2
    cell_mask = label_xy > 0

    pd1_xy = xydata[..., 1+index_pd1]

    # generate scenarios
    selem = morphology.disk(1) # disk with diameter of 1 pixel
    fn_dict = {
            'e': lambda arr: morphology.binary_erosion(arr, selem=selem),
            'x': lambda arr: arr,
            'd': lambda arr: morphology.binary_dilation(arr, selem=selem),
            }
    for nuclei_op in fn_dict:
        for cell_op in fn_dict:
            nuclei_fn = fn_dict[nuclei_op]
            cell_fn = fn_dict[cell_op]
            nm = nuclei_fn(nuclei_mask)
            cm = cell_fn(cell_mask)
            lxy = nm.astype(int) + cm.astype(int)
            scenario = np.stack([lxy, pd1_xy], axis=-1)
            output_filepath = 'scenario_data/radial_n{}c{}.npy'\
                    .format(nuclei_op, cell_op)
            np.save(output_filepath, scenario)

    print('all done.')
