import os
import sys
code_folderpath = os.path.expanduser('~/polartk/code/')
sys.path.append(code_folderpath)

import numpy as np
import matplotlib.pyplot as plt

import polartk

from scipy import stats, optimize, interpolate
from sklearn import neighbors

def infer_centroid(image, num_bin=10):
    x, y = np.meshgrid(range(image.shape[0]), range(image.shape[1]),
            indexing='ij')

    def loss(p):
        r = np.sqrt((x-p[0])**2 + (y-p[1])**2)
        r_grid = np.unique(r)
        np.sort(r_grid)

        r_bins = np.array_split(r_grid, num_bin)
        bin_std = np.zeros(num_bin)

        for i in range(num_bin):
            bin_low, bin_high = r_bins[i].min(), r_bins[i].max()
            mask = (r >= bin_low) & (r < bin_high)
            bin_std[i] = image[mask].std()

        return bin_std.mean()

    opt_result = optimize.minimize(fun=loss, method='Nelder-Mead',
            x0=(np.median(x), np.median(y)))
    if opt_result.success:
        return opt_result.x
    else:
        print(opt_result.message)
        return opt_result.x

def infer_stretch(z_rt):
    angular_abun = z_rt.sum(axis=0)
    anchor_index = np.argmin(angular_abun)
    k_rt = np.zeros_like(z_rt)

    for t_index in range(z_rt.shape[1]):
        ref_profile = z_rt[:, anchor_index]
        this_profile = z_rt[:, t_index]

        r_grid = np.arange(z_rt.shape[0])
        z_grid = np.linspace(
                max(ref_profile.min(), this_profile.min()),
                min(ref_profile.max(), this_profile.max()),
                num=100)
        ref_r_pred = interpolate.interp1d(ref_profile, r_grid)(z_grid)
        this_r_pred = interpolate.interp1d(this_profile, r_grid)(z_grid)

        factor = this_r_pred / ref_r_pred
        mask = np.isfinite(factor)
        factor_model = interpolate.interp1d(z_grid[mask], factor[mask])
        within_range = (this_profile > z_grid[mask].min())\
                & (this_profile < z_grid[mask].max())
        k_rt[within_range, t_index] = factor_model(this_profile[within_range])

    return k_rt

def xy2rt_maskfree(dna_xy, image_xy_list, r_threshold=None):
    # infer centroid
    x, y = np.meshgrid(range(dna_xy.shape[0]),
            range(dna_xy.shape[1]),
            indexing='ij')
    xc_est, yc_est = infer_centroid(dna_xy)

    # do standard polar coordinate transformation
    out_dict = polartk.xy2rt(images=[dna_xy], centroid=(xc_est, yc_est))
    z_rt = out_dict['image_rt_list'][0]

    # infer radius scaling factor
    k_rt = infer_stretch(z_rt)
    if r_threshold is not None:
        r_ladder = out_dict['r_out'][:, 0]
        k_rt[r_ladder > r_threshold, :] = 1 # mask out neighbors for radius estimation

    # prepare grids for modeling
    rt_out_grid = np.stack([out_dict['r_out'].flatten(),
        out_dict['t_out'].flatten()], axis=-1)
    r_in = out_dict['r_in']#[1:-1, 1:-1] # remove padding
    t_in = out_dict['t_in']#[1:-1, 1:-1] # remove padding
    rt_in_grid = np.stack([r_in.flatten(), t_in.flatten()], axis=-1)

    # transform radius scaling factor from polar coordinate to Euclidean coordinate
    model = neighbors.KNeighborsRegressor(metric=polartk.polar_dist)
    model.fit(rt_out_grid, k_rt.flatten())
    k_xy = model.predict(rt_in_grid).reshape(r_in.shape)

    # crop factors for numerical stability and calculate effective radius
    k_xy[k_xy == 0] = 1
    r_in_eff = np.divide(r_in, k_xy)

    # get new grid
    rt_in_grid_eff = np.stack([r_in_eff.flatten(), t_in.flatten()], axis=-1)

    # transform DNA
    dna_xy_pad = np.pad(dna_xy, pad_width=1, constant_values=0)
    model = neighbors.KNeighborsRegressor(metric=polartk.polar_dist)
    model.fit(rt_in_grid_eff, dna_xy_pad.flatten())
    dna_rt = model.predict(rt_out_grid).reshape(dna_xy.shape)

    # transform other images
    image_rt_list = []
    for image_xy in image_xy_list:
        image_xy_pad = np.pad(image_xy, pad_width=1, constant_values=0)
        model = neighbors.KNeighborsRegressor(metric=polartk.polar_dist)
        model.fit(rt_in_grid_eff, image_xy_pad.flatten())
        image_rt = model.predict(rt_out_grid).reshape(image_xy.shape)
        image_rt_list.append(image_rt)

    new_out_dict = {
            'x_in': x, 'y_in': y, 'r_in': r_in_eff, 't_in': t_in,
            'r_out': out_dict['r_out'], 't_out': out_dict['t_out'],
            'k_in': k_xy, 'DNA_rt': dna_rt, 'image_rt_list': image_rt_list}
    return new_out_dict

if __name__ == '__main__':
    # load data
    job = np.load('job_1497.npy')
    label_xy = job[..., 0]
    image_list = [job[..., i] for i in range(1, job.shape[2])]

    # standard polar coordinate transform
    out_dict_std = polartk.xy2rt(images=image_list)
    out_std = np.stack(out_dict_std['image_rt_list'], axis=-1)

    # mask-based transformation
    out_dict_mask = polartk.xy2rt(images=image_list, label=label_xy)
    out_mask = np.stack(out_dict_mask['image_rt_list'], axis=-1)

    # mask-free transformation
    dna_xy = job[..., 1]
    out_dict_maskfree = xy2rt_maskfree(dna_xy=dna_xy, image_xy_list=image_list,
            r_threshold=8)
    out_maskfree = np.stack(out_dict_maskfree['image_rt_list'], axis=-1)

    np.save('im_xy.npy', job[..., 1:])
    np.save('im_rt_std.npy', out_std)
    np.save('im_rt_mask.npy', out_mask)
    np.save('im_rt_maskfree.npy', out_maskfree)

