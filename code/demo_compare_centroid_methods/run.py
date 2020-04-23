import os

import numpy as np
import matplotlib.pyplot as plt

from scipy  import stats
from skimage.external import tifffile
from skimage import io, exposure, segmentation, feature

if __name__ == '__main__':
    # paths
    data_folderpath = os.path.expanduser('~/polar_data/data')
    image_filepath = os.path.join(data_folderpath, 'image.ome.tif')
    nuclei_mask_filepath = os.path.join(data_folderpath, 'nuclei_mask.tif')
    unet_pm_filepath = os.path.join(data_folderpath, 'unet_NucleiPM.tif')

    # region of interest
    select_roi = lambda arr: arr[1320:1380, 240:290]

    # load data
    with tifffile.TiffFile(image_filepath) as infile:
        dna = infile.series[0].pages[0].asarray()

    nuclei_mask = io.imread(nuclei_mask_filepath)
    unet_pm = io.imread(unet_pm_filepath)[0, ...]

    # preprocessing
    dna = dna.astype(float)
    dna = exposure.rescale_intensity(dna, out_range=(0, 1),
            in_range=tuple(np.percentile(dna, (1, 99))))

    nuclei_outline = segmentation.find_boundaries(nuclei_mask > 0, mode='inner')
    unet_pm = unet_pm.astype(float)
    unet_pm = exposure.rescale_intensity(unet_pm, out_range=(0, 1))

    # template matching
    x, y = np.meshgrid(range(15), range(15), indexing='ij')
    xy = np.stack([x.flatten(), y.flatten()], axis=-1)
    z = stats.multivariate_normal(mean=np.median(xy, axis=0), cov=(15, 15))\
            .pdf(xy).reshape(x.shape)
    response = feature.match_template(dna, z, pad_input=True)
    response = exposure.rescale_intensity(response, out_range=(0, 1))

    # plot
    fig, axes = plt.subplots(ncols=3, nrows=2, sharex=True, sharey=True,
            figsize=(9, 6))

    im_list = [dna, unet_pm, response]

    # get ROI
    im_list = [select_roi(im) for im in im_list]
    nuclei_outline = select_roi(nuclei_outline)

    for row_index in range(2):
        for col_index in range(3):

            im = im_list[col_index]
            rgb = np.stack([np.zeros_like(im), im, np.zeros_like(im)], axis=-1)

            if row_index == 1:
                for ch in range(rgb.shape[2]):
                    rgb[..., ch] = np.maximum(rgb[..., ch], nuclei_outline)

            axes[row_index, col_index].imshow(rgb)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_title('DNA1 intensity')
    axes[0, 1].set_title('Unet prob maps')
    axes[0, 2].set_title('template matching')

    axes[0, 0].set_ylabel('image')
    axes[1, 0].set_ylabel('image + segmentation outline')

    fig.tight_layout()
    plt.savefig(os.path.expanduser('~/polartk/figures/'\
            'demo_compare_centroid_methods.png'), dpi=600)
    plt.show()
