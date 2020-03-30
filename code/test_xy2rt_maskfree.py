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

def xy2rt_maskfree(z):
    x, y = np.meshgrid(range(z.shape[0]), range(z.shape[1]), indexing='ij')
    xc_est, yc_est = infer_centroid(z)

    out_dict = polartk.xy2rt(images=[z], centroid=(xc_est, yc_est))
    z_rt = out_dict['image_rt_list'][0]

    k_rt = infer_stretch(z_rt)

    rt_out_grid = np.stack([out_dict['r_out'].flatten(),
        out_dict['t_out'].flatten()], axis=-1)
    r_in = out_dict['r_in'][1:-1, 1:-1] # remove padding
    t_in = out_dict['t_in'][1:-1, 1:-1] # remove padding
    rt_in_grid = np.stack([r_in.flatten(), t_in.flatten()], axis=-1)

    model = neighbors.KNeighborsRegressor(metric=polartk.polar_dist)
    model.fit(rt_out_grid, k_rt.flatten())
    k_xy = model.predict(rt_in_grid).reshape(z.shape)

    k_xy[k_xy == 0] = 1
    r_in_eff = np.divide(r_in, k_xy)

    rt_in_grid_eff = np.stack([r_in_eff.flatten(), t_in.flatten()], axis=-1)
    model = neighbors.KNeighborsRegressor(metric=polartk.polar_dist)
    model.fit(rt_in_grid_eff, z.flatten())
    z_rt_eff = model.predict(rt_out_grid).reshape(z_rt.shape)

    return z_rt_eff

if __name__ == '__main__':
    # build problem
    scope = (30, 30)
    x, y = np.meshgrid(range(scope[0]), range(scope[1]),
            indexing='ij')
    xy = np.stack([x.flatten(), y.flatten()], axis=-1)
    xc, yc = 14.5, 14.5
    z = stats.multivariate_normal(mean=(xc, yc), cov=(20, 7))\
            .pdf(xy).reshape(scope)
    z /= z.max()

    # add noise
    noise = np.random.randn(*z.shape).reshape(z.shape) * 0.05
    z += noise
    z -= z.min()
    z /= z.max()

    # standard polar coordinate transform
    out_dict = polartk.xy2rt(images=[z])
    z_rt = out_dict['image_rt_list'][0]

    z_rt_eff = xy2rt_maskfree(z)

    params = dict(cmap='gray',
            vmin=min(z.min(), z_rt.min(), z_rt_eff.min()),
            vmax=max(z.max(), z_rt.max(), z_rt_eff.max()),
            )
    fig, axes = plt.subplots(ncols=3, nrows=1, sharex=True, sharey=True,
            figsize=(9, 3))
    axes[0].imshow(z, **params)
    axes[0].set_title('Euclidean coordinate')

    axes[1].imshow(z_rt, **params)
    axes[1].set_title('previous way\nof transformation')

    axes[2].imshow(z_rt_eff, **params)
    axes[2].set_title('new way\nof transformation')

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()

