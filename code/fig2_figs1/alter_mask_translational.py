import os

import numpy as np

from skimage import transform

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
    shift_list = [-1, 0, 1]
    for shift_x in shift_list:
        for shift_y in shift_list:
            t_fn = transform.AffineTransform(translation=(-shift_x, -shift_y))
            cm = transform.warp(cell_mask, t_fn.inverse)
            lxy = nuclei_mask.astype(int) + cm.astype(int)
            scenario = np.stack([lxy, pd1_xy], axis=-1)
            output_filepath = 'scenario_data/translational_c{}{}.npy'\
                    .format(shift_x, shift_y)
            np.save(output_filepath, scenario)

    print('all done.')
