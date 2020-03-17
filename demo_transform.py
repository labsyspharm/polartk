import numpy as np
import matplotlib.pyplot as plt

from skimage import segmentation
from scipy import stats

import polartk

if __name__ == '__main__':
    # params
    tile_shape = (15, 15)
    
    # x, y coordinate grid
    x, y = np.meshgrid(range(tile_shape[0]), range(tile_shape[1]),
                       indexing='ij')
    xc, yc = np.median(x), np.median(y)
    
    # nuclei and cytoplasm
    xy = np.stack([x.flatten(), y.flatten()], axis=-1)
    
    nuclei = stats.multivariate_normal(mean=(xc, yc), cov=((2.5, 1), (1, 2.5)))\
        .pdf(xy).reshape(tile_shape)
    nuclei /= nuclei.max()
    
    cell = stats.multivariate_normal(mean=(xc, yc), cov=((10, 5), (5, 10)))\
        .pdf(xy).reshape(tile_shape)
    cell /= cell.max()
    cyto = cell - nuclei
    
    # construct label
    label_xy = np.zeros(tile_shape)
    label_xy[cell > 0.1] = 1
    label_xy[nuclei > 0.1] = 2
    
    _, _, nuclei_rt = polartk.xy2rt(image=nuclei, centroid=(xc, yc))
    _, _, cyto_rt = polartk.xy2rt(image=cyto, centroid=(xc, yc))
    
    _, _, label_rt = polartk.xy2rt(image=label_xy, centroid=(xc, yc))
    label_rt = np.round(label_rt).astype(int)
    label_rt[label_rt < 0] = 0
    label_rt[label_rt > 2] = 2
    
    _, _, nuclei_relrt, label_relrt = polartk.xy2rt(image=nuclei, centroid=(xc, yc), label=label_xy)
    _, _, cyto_relrt, _ = polartk.xy2rt(image=cyto, centroid=(xc, yc), label=label_xy)
    
    fig, axes = plt.subplots(ncols=3, nrows=2, sharex=True, sharey=True, figsize=(12, 8))
    
    label_cmap = 'tab10'
    axes[0, 0].imshow(label_xy, cmap=label_cmap)
    axes[0, 0].scatter([yc], [xc], color='red', marker='o')
    axes[0, 1].imshow(label_rt, cmap=label_cmap)
    axes[0, 2].imshow(label_relrt, cmap=label_cmap)
    
    rgb = np.stack([nuclei, cyto, np.zeros(tile_shape)], axis=-1)
    rgb[..., 1] += nuclei
    rgb[..., 2] += nuclei
    axes[1, 0].imshow(rgb)
    
    rgb = np.stack([nuclei_rt, cyto_rt, np.zeros(tile_shape)], axis=-1)
    rgb[..., 1] += nuclei_rt
    rgb[..., 2] += nuclei_rt
    axes[1, 1].imshow(rgb)
    
    rgb = np.stack([nuclei_relrt, cyto_relrt, np.zeros(tile_shape)], axis=-1)
    rgb[..., 1] += nuclei_relrt
    rgb[..., 2] += nuclei_relrt
    axes[1, 2].imshow(rgb)
        
    fs = 16
    axes[0, 0].set_title('xy-coordinate', fontsize=fs)
    axes[0, 1].set_title('abs. polar coordinate', fontsize=fs)
    axes[0, 2].set_title('rel. polar coordinate', fontsize=fs)
    
    axes[0, 0].set_ylabel('label', fontsize=fs)
    axes[1, 0].set_ylabel('image', fontsize=fs)
        
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.suptitle('Demo absolute vs. relative polar coordinate', fontsize=fs)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    