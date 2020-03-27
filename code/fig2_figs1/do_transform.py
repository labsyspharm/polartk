import os
import sys
import typing

import numpy as np
import tqdm

from scipy.ndimage import morphology
from sklearn import neighbors

def polar_dist(
    a: typing.Tuple[float, float],
    b: typing.Tuple[float, float],
    ):
    '''
    Distance metric in polar coordinate.
    
    Args
        a, b: tuple of float
            First element is radius. Second is angle in radian.
        
    Return
        distance: float
    '''
    r1, t1 = a
    r2, t2 = b
    return np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * np.cos(t1 - t2))

def xy2rt(job: np.ndarray) -> np.ndarray:
    # params
    params = {
        'pw': 1, # pad width, unit: pixel
        'pv': 0, # pad value for image
        'n_neighbors': 5, # params for KNN intensity model
    }

    # pad to remove boundary conditions
    label_xy = job[..., 0]
    label_xy_pad = np.pad(label_xy, pad_width=params['pw'], mode='constant',
                       constant_values=0) # 0=background
        
    # prepare euclidean coordinate
    x, y = np.meshgrid(
            np.arange(label_xy_pad.shape[0]),
            np.arange(label_xy_pad.shape[1]),
            indexing='ij')
    nuclei_pixels = np.argwhere(label_xy_pad == 2)
    xc, yc = np.mean(nuclei_pixels, axis=0)
    
    # prepare polar coordinate
    r_nuclei = morphology.distance_transform_edt(label_xy_pad == 2)
    r_nuclei = r_nuclei.max() - r_nuclei
    r_cell = morphology.distance_transform_edt(label_xy_pad < 2)
    r_background = morphology.distance_transform_edt(label_xy_pad == 0)
    r = r_nuclei + r_cell + r_background
    t = np.arctan2(x-xc, y-yc)
    
    # prepare grid points after transformation
    r_grid, t_grid = np.meshgrid(
            np.linspace(start=0, stop=r.max(), num=job.shape[0]),
            np.linspace(start=-np.pi, stop=np.pi, num=job.shape[1], endpoint=False),
            indexing='ij')

    # approximate label by KNN
    rt = np.stack([r.flatten(), t.flatten()], axis=-1)
    rt_grid = np.stack([r_grid.flatten(), t_grid.flatten()], axis=-1)
    output_array = np.zeros_like(job)

    label_model = neighbors.KNeighborsClassifier(metric=polar_dist,
            n_neighbors=1)
    label_model.fit(rt, label_xy_pad.flatten())
    label_rt  = label_model.predict(rt_grid)
    output_array[..., 0] = label_rt.reshape(job.shape[:-1])

    # approximate intensities by KNN
    intensity_model = neighbors.KNeighborsRegressor(metric=polar_dist,
        n_neighbors=params['n_neighbors'])
    for i in range(1, job.shape[2]):
        image_pad = np.pad(job[..., i], pad_width=params['pw'], mode='constant',
            constant_values=params['pv'])
        intensity_model.fit(rt, image_pad.flatten())
        image_rt = intensity_model.predict(rt_grid)
        output_array[..., i] = image_rt.reshape(job.shape[:-1])

    return output_array

if __name__ == '__main__':
    scenario_list = [name for name in os.listdir('.') if name.startswith('scenario_')]
    for input_filepath in tqdm.tqdm(scenario_list):
        job = np.load(input_filepath)
        output = xy2rt(job)
        np.save('output_' + input_filepath, output)

