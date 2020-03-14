import numpy as np
from scipy.ndimage import morphology
from sklearn import neighbors

# for main function
import matplotlib.pyplot as plt
from skimage import measure, io, exposure
from skimage.external import tifffile

def xy2rt_v1(im, n_neighbors=5, new_shape=None, rel_distance=False, mask=None):
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

def xy2rt_v2(intensity_image, cell_centroid, nuclei_mask, cell_mask,
        n_neighbors=5, new_shape=None):
    # preparation
    def polar_dist(a, b):
        '''
        distance metric in polar coordinate
        '''
        r1, t1 = a
        r2, t2 = b
        return np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * np.cos(t1 - t2))

    # label, 0=background, 1=cytoplasm, 2=nuclei
    label_xy = np.zeros_like(cell_mask, dtype=int)
    label_xy[cell_mask] = 1
    label_xy[nuclei_mask] = 2

    # pad to remove boundary conditions
    pw = 1 # 1 pixel
    pv = 0 # 0=background
    label_xy_pad = np.pad(label_xy, pad_width=pw, mode='constant', constant_values=pv)
    intensity_image_pad = np.pad(intensity_image, pad_width=pw, mode='constant',
            constant_values=pv)

    # x, y, r (radius), theta (angle in radian)
    x, y = np.meshgrid(
            np.arange(label_xy_pad.shape[0]),
            np.arange(label_xy_pad.shape[1]),
            indexing='ij')
    xc, yc = cell_centroid[0]+pw, cell_centroid[1]+pw
    r_nuclei = morphology.distance_transform_edt(label_xy_pad == 2)
    r_nuclei = r_nuclei.max() - r_nuclei
    r_cell = morphology.distance_transform_edt(label_xy_pad < 2)
    r_background = morphology.distance_transform_edt(label_xy_pad == 0)
    r = r_nuclei + r_cell + r_background
    t = np.arctan2(x-xc, y-yc)
    rt = np.stack([r.flatten(), t.flatten()], axis=-1)

    # classification model: (R, Theta) --> label
    label_model = neighbors.KNeighborsClassifier(n_neighbors=1, metric=polar_dist)
    label_model.fit(rt, label_xy_pad.flatten())
    intensity_model = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors,
            metric=polar_dist)
    intensity_model.fit(rt, intensity_image_pad.flatten())

    # create (R, Theta) grid
    # note that angle 2*pi == 0, so endpoint=False for angle
    if new_shape is None:
        new_shape = intensity_image_pad.shape
    r_grid, t_grid = np.meshgrid(
            np.linspace(start=r.min(), stop=r.max(), num=new_shape[0]),
            np.linspace(start=0, stop=2*np.pi, num=new_shape[1], endpoint=False),
            indexing='ij')
    rt_grid = np.stack([r_grid.flatten(), t_grid.flatten()], axis=-1)

    # predict region and intensity
    label_rt = label_model.predict(rt_grid).reshape(new_shape)
    intensity_rt = intensity_model.predict(rt_grid).reshape(new_shape)

    return label_rt, intensity_rt

if __name__ == '__main__':
    # load data
    cell_mask = io.imread('../data/seg_masks/ZM131_10B_286_roi_A_cellRingMask.tif')
    nuclei_mask = io.imread('../data/seg_masks/ZM131_10B_286_roi_A_nucleiRingMask.tif')
    with tifffile.TiffFile('../data/images/ZM131_10B_286_roi_A_masked.ome.tif') as tif:
        dna = tif.series[0].pages[0].asarray()
        ps6 = tif.series[0].pages[21].asarray()

    # get region
    target = 6640
    nuclei_region_list = measure.regionprops(label_image=nuclei_mask)
    nuclei_region = [region for region in nuclei_region_list if region.label == target][0]
    cell_region_list = measure.regionprops(label_image=cell_mask)
    cell_region = [region for region in cell_region_list if region.label == target][0]

    # unify coordinate to cell bounding box
    cxl, cyl, cxu, cyu = cell_region.bbox
    cm = cell_region.image
    # nuclei
    nxl, nyl, nxu, nyu = nuclei_region.bbox
    nm = np.zeros_like(cm)
    nm[(nxl-cxl):(nxu-cxl), (nyl-cyl):(nyu-cyl)] = nuclei_region.image
    # cell centroid
    cc = nuclei_region.centroid
    cc = (cc[0]-cxl, cc[1]-cyl)
    # markers
    dna_i = dna[cxl:cxu, cyl:cyu]
    ps6_i = ps6[cxl:cxu, cyl:cyu]
    
    # test
    label_rt, dna_rt = xy2rt_v2(intensity_image=dna_i, cell_centroid=cc, nuclei_mask=nm,
            cell_mask=cm)
    _, ps6_rt = xy2rt_v2(intensity_image=ps6_i, cell_centroid=cc, nuclei_mask=nm,
            cell_mask=cm)

    rgb_xy = np.zeros(dna_i.shape + (3,), dtype=float)
    rgb_xy[..., 0] = exposure.rescale_intensity(dna_i.astype(float),
            out_range=(0, 1),
            in_range=tuple(np.percentile(dna_i, (1, 99))))
    rgb_xy[..., 1] = exposure.rescale_intensity(ps6_i.astype(float),
            out_range=(0, 1),
            in_range=tuple(np.percentile(ps6_i, (1, 99))))

    rgb_rt = np.zeros(dna_rt.shape + (3,), dtype=float)
    rgb_rt[..., 0] = exposure.rescale_intensity(dna_rt.astype(float),
            out_range=(0, 1),
            in_range=tuple(np.percentile(dna_rt, (1, 99))))
    rgb_rt[..., 1] = exposure.rescale_intensity(ps6_rt.astype(float),
            out_range=(0, 1),
            in_range=tuple(np.percentile(ps6_rt, (1, 99))))

    plt.subplot(121); plt.imshow(rgb_xy); plt.title('xy')
    plt.subplot(122); plt.imshow(rgb_rt); plt.title('rt')
    plt.tight_layout(); plt.show()
