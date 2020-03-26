import typing

import numpy as np

from scipy.ndimage import morphology
from sklearn import neighbors, preprocessing
from scipy import stats

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

def xy2rt(
    image: np.ndarray,
    centroid: typing.Sequence[float]=None,
    label: np.ndarray=None,
    shape: typing.Tuple[int, int]=None,
    params: typing.Dict[str, typing.Union[int, float]]=None,
    ) -> typing.Sequence[np.ndarray]:
    '''
    Transform an image from xy (Euclidean) coordinate to rt (polar) coordinate.
    
    Args
        image: numpy.ndarray
            The intensity image to be transformed.
        centroid: sequence of float, optional
            Centroid, in xy coordinate, within the image to be used as the
            origin of the polar coordinate. If not specified, the median
            of the image dimension will be used.
        label: numpy.ndarray, optional
            Label (mask) of the image, 0=background, 1=cytoplasm, 2=nuclei.
            If specified, the radius of the polar coordinate will be the relative
            distance from the cell contour. If not specified, the radius will
            be relative to the centroid.
        shape: tuple of int, optional
            Output shape. Used to generate the output grid for prediction.
            If not specified, the shape of input ("image") will be used.
            Can be larger than the shape of input, but depending on the
            input resolution and application, the output resolution may
            or may not make sense.
        params: dict of str:number pairs, optional
            Parameters, including:
            pw: pad width, unit: pixel
            pv: pad value for image
            n_neighbors: params for KNN intensity model
            
    Return
        r_grid: np.ndarray
            Radius grid of the output, unit: pixel.
        t_grid: np.ndarray
            Angle grid of the output, unit: radian.
        image_rt: np.ndarray
            Transformed image.
        label_rt: np.ndarray, if label is given
            Transformed label.
    '''
    if params is None:
        params = {
            'pw': 1, # pad width, unit: pixel
            'pv': 0, # pad value for image
            'n_neighbors': 5, # params for KNN intensity model
        }
    # pad to remove boundary conditions
    image_pad = np.pad(image, pad_width=params['pw'], mode='constant',
            constant_values=params['pv'])
    if label is not None:
        label_pad = np.pad(label, pad_width=params['pw'], mode='constant',
                           constant_values=0) # 0=background
        
    # x, y
    x, y = np.meshgrid(
            np.arange(image_pad.shape[0]),
            np.arange(image_pad.shape[1]),
            indexing='ij')
    if centroid is None:
        xc, yc = np.median(x), np.median(y)
    else:
        xc, yc = centroid
    
    # radius
    if label is None:
        r = np.sqrt((x-xc)**2 + (y-yc)**2)
    else:
        r_nuclei = morphology.distance_transform_edt(label_pad == 2)
        r_nuclei = r_nuclei.max() - r_nuclei
        r_cell = morphology.distance_transform_edt(label_pad < 2)
        r_background = morphology.distance_transform_edt(label_pad == 0)
        r = r_nuclei + r_cell + r_background
        
    # angle (radian)
    t = np.arctan2(x-xc, y-yc)
    
    # approximate with KNN
    rt = np.stack([r.flatten(), t.flatten()], axis=-1)
    intensity_model = neighbors.KNeighborsRegressor(metric=polar_dist,
        n_neighbors=params['n_neighbors'])
    intensity_model.fit(rt, image_pad.flatten())
    if label is not None:
        label_model = neighbors.KNeighborsClassifier(metric=polar_dist,
                n_neighbors=1)
        label_model.fit(rt, label_pad.flatten())

    # create (R, Theta) grid
    # note that angle 2*pi == 0, so endpoint=False for angle
    if shape is None:
        shape = image.shape
    r_grid, t_grid = np.meshgrid(
            np.linspace(start=0, stop=r.max(), num=shape[0]),
            np.linspace(start=-np.pi, stop=np.pi, num=shape[1], endpoint=False),
            indexing='ij')
    
    # predict region and intensity
    rt_grid = np.stack([r_grid.flatten(), t_grid.flatten()], axis=-1)
    image_rt = intensity_model.predict(rt_grid).reshape(shape)
    if label is not None:
        label_rt = label_model.predict(rt_grid).reshape(shape)
    
    if label is None:
        return r_grid, t_grid, image_rt
    else:
        return r_grid, t_grid, image_rt, label_rt
    
def polarity(d: typing.Sequence[float]):
    '''
    Custom definition of polarity.
    
    polarity := s/(s+1)
    
    s := cross entropy between the given angular distribution of intensity
    and an uniform distribution.
    
    Cross entropy has range of [0, inf), and polarity defined here has
    range of [0, 1). The transformation s/(s+1) is to make the output have
    a more intuitive range.
    
    Args
        d: sequence of float
            Intensity at different angle (angular distribution).
            
    Return
        polarity: float
    '''
    # handle nan
    d = d[np.isfinite(d)]
    # maek reference
    dref = np.ones_like(d)
    dref /= dref.sum()
    # handle numerical stability
    d2 = d + 1e-6
    d2 /= d2.sum()
    # polarity = s/(s+1), s = cross entropy
    s = stats.entropy(dref, d2) # range [0, inf)
    p = s/(s+1) # range [0, 1)
    return p
