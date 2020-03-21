import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from scipy import interpolate
from skimage.external import tifffile
from skimage import measure, io, exposure
from matplotlib.legend_handler import HandlerPatch

class HandlerArrow(HandlerPatch):
    '''
    For showing an arrow in a custom legend.

    Source:
    https://stackoverflow.com/questions/60781312/plotting-arrow-in-front-of-legend-matplotlib
    '''
    def __init__(self, arrow_params):
        super().__init__()
        self.arrow_params = arrow_params

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width,
            height, fontsize, trans):
        p = mpatches.FancyArrow(0, 0.5*height, width, 0,
                head_width=0.75*height, **self.arrow_params)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

def prep_intensity(arr):
    '''
    function to stretch contrast
    '''
    arr = arr.astype(float)
    arr = exposure.rescale_intensity(arr, out_range=(0, 1),
            in_range=tuple(np.percentile(arr, (1, 99))))
    return arr

if __name__ == '__main__':
    # paths
    data_folderpath = os.path.expanduser('~/polar_data')
    image_filepath = os.path.join(data_folderpath, 'data', 'image.ome.tif')
    nuclei_mask_filepath = os.path.join(data_folderpath, 'data', 'nuclei_mask.tif')

    talign_filepath = os.path.join(data_folderpath, 'pd1_angular_alignment.csv')

    # params
    view_shape = (100, 100)
    tile_shape = (15, 15)

    # load data
    with tifffile.TiffFile(image_filepath) as infile:
        dna = infile.series[0].pages[0].asarray()
        sox10 = infile.series[0].pages[7].asarray()
        pd1 = infile.series[0].pages[19].asarray()

    dna = prep_intensity(dna)
    sox10 = prep_intensity(sox10)
    pd1 = prep_intensity(pd1)

    nuclei_mask = io.imread(nuclei_mask_filepath)
    talign_df = pd.read_csv(talign_filepath)

    region_dict = {r.label:r for r in measure.regionprops(nuclei_mask)}
    talign_df['xc'] = talign_df['cellid'].apply(
            lambda x: region_dict[x].centroid[0])
    talign_df['yc'] = talign_df['cellid'].apply(
            lambda x: region_dict[x].centroid[1])

    rgb_image = np.stack([sox10, pd1, np.zeros_like(dna)], axis=-1)
    for c in range(3):
        rgb_image[..., c] = np.maximum(rgb_image[..., c], dna*0.5)

    talign_df.sort_values('polarity', ascending=False, inplace=True)
    talign_df.reset_index(drop=True, inplace=True)

    # plot both positive and negative example
    # examples picked through visual inspection
    pos_cellid = 8069
    neg_cellid = 7886

    for cellid in [pos_cellid, neg_cellid]:
        index = talign_df.index[talign_df['cellid'] == cellid][0]
        row = talign_df.loc[index]
        # construct vector
        r = 5 * row['polarity']
        t = row['angle']
        dx = r * np.cos(np.radians(t))
        dy = r * np.sin(np.radians(t))
        # select image
        region = region_dict[row['cellid']]
        xc, yc = region.centroid
        xl = int(np.round(xc-view_shape[0]/2))
        yl = int(np.round(yc-view_shape[1]/2))
        xu, yu = xl+view_shape[0], yl+view_shape[1]
        # check border
        valid_list = [xl >= 0, yl >= 0, xu < dna.shape[1], yu < dna.shape[0]]
        if not all(valid_list):
            continue

        # construct figure
        txl = int(np.round(xc-tile_shape[0]/2))
        tyl = int(np.round(yc-tile_shape[1]/2))
        txu, tyu = txl+tile_shape[0], tyl+tile_shape[1]

        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10,5))
        axes[0].imshow(rgb_image[xl:xu, yl:yu])
        axes[0].set_title('full view')
        tile_patch_1 = mpatches.Rectangle(xy=(tyl-yl, txl-xl),
                width=tile_shape[1]-1, height=tile_shape[0]-1,
                facecolor='none', edgecolor='magenta', linestyle='dashed')
        axes[0].add_patch(tile_patch_1)

        axes[1].imshow(rgb_image[txl:txu, tyl:tyu])
        arrow_params = dict(color='magenta', overhang=0.5, length_includes_head=True)
        axes[1].arrow(yc-tyl-dx, xc-txl-dy, 2*dx, 2*dy,
                width=0.05, head_length=1, head_width=1,
                **arrow_params,
                label='polarization vector')
        tile_patch_2 = mpatches.Rectangle(xy=(0, 0),
                width=tile_shape[1]-1, height=tile_shape[0]-1,
                facecolor='none', edgecolor='magenta', linestyle='dashed')
        axes[1].add_patch(tile_patch_2)
        axes[1].set_title('zoom-in')

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        legend_elements = [
                mpatches.Patch(facecolor='red', edgecolor='red',
                    label='SOX10'),
                mpatches.Patch(facecolor='lime', edgecolor='lime',
                    label='PD1'),
                mpatches.Patch(facecolor='white', edgecolor='white',
                    label='DNA'),
                mpatches.Patch(facecolor='none', edgecolor='magenta',
                    linestyle='dashed', label=r'9.75x9.75 $\mu$m'),
                mpatches.FancyArrow(x=0, y=0, dx=0, dy=0, width=3,
                    label='polarization vector', **arrow_params),
                ]
        L = plt.figlegend(handles=legend_elements,
                bbox_to_anchor=(0.95, 0.99), loc='upper right',
                facecolor='black', framealpha=1, ncol=5,
                handler_map={mpatches.FancyArrow : HandlerArrow(arrow_params)},
                )
        for t in L.get_texts():
            t.set_color('white')

        fig.suptitle('cell ID: {:.0f}'.format(row['cellid']), x=0.2, y=0.96)
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        plt.savefig('../figures/demo_vector_cellid_{:.0f}.png'.format(row['cellid']))
        plt.show()
        plt.close()
