import numpy as np

from skimage import morphology
from sklearn import neighbors

def iter_cell(cellid_array):
    '''
    Iterate over (x,y) coordinates given a cell ID mask.
    '''
    # flatten
    cellid = cellid_array.flatten()
    pixelid = np.arange(cellid.shape[0], dtype=int)
    # by convention, background has value of zero
    mask = cellid > 0
    cellid, pixelid = cellid[mask], pixelid[mask]
    # sort by cell ID
    sortkey = np.argsort(cellid)
    cellid, pixelid = cellid[sortkey], pixelid[sortkey]
    unique_cellid, count_cellid = np.unique(cellid, return_counts=True)
    cut_point = np.cumsum(count_cellid)
    sequence = zip(unique_cellid, np.split(pixelid, cut_point))
    # store
    cellcoord = []
    for ci, pi in sequence:
        xi, yi = np.unravel_index(pi, cellid_array.shape)
        yield ci, xi, yi

def xy2rt(im, n_neighbors=5, new_shape=None, rel_distance=False, mask=None):
    if new_shape is None:
        new_shape = im.shape
    # construct r (radius) and t (angle) grid
    x, y = np.meshgrid(range(im.shape[0]), range(im.shape[1]), indexing='xy')
    xc, yc = np.median(x), np.median(y)
    if rel_distance:
        _, r_inside = morphology.medial_axis(mask, return_distance=True)
        r_inside = r_inside.max() - r_inside
        _, r_outside = morphology.medial_axis(np.logical_not(mask), return_distance=True)
        r = r_inside + r_outside
    else:
        r = np.sqrt((x-xc)**2 + (y-yc)**2)
    t = np.arctan2(x-xc, y-yc)
    def polar_dist(a, b):
        r1, t1 = a
        r2, t2 = b
        return np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * np.cos(t1 - t2))
    # KNN is more sensitive in dense area
    # approximate by KNN regression
    rt = np.stack([r.flatten(), t.flatten()], axis=-1)
    model = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors,
            metric=polar_dist)
    model.fit(rt, im.flatten())
    # evaluate on new grid
    new_r, new_t = np.meshgrid(
            np.linspace(r.min(), r.max(), new_shape[0]),
            np.linspace(t.min(), t.max(), new_shape[1]),
            indexing='xy')
    new_rt = np.stack([new_r.flatten(), new_t.flatten()], axis=-1)
    new_im = model.predict(new_rt)
    new_im = new_im.reshape(new_shape)
    return new_im

