import os

import numpy as np
import matplotlib.pyplot as plt

def rcross_corrcoef(a, b):
    cc = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        a_small = a[0:a.shape[0]-i, :]
        b_small = b[i:a.shape[0], :]
        cc[i] = np.corrcoef(a_small.flatten(), b_small.flatten())[0, 1]
    return cc

def corr_metric(a, b):
    inv_a = a.max(axis=1, keepdims=True) - a
    response = rcross_corrcoef(a, b)
    inv_response = rcross_corrcoef(inv_a, b)
    return response - inv_response

if __name__ == '__main__':
    # paths
    output_folderpath = os.path.expanduser('~/polartk/figures/')
    input_filepath = 'demo2_rgb_rt.npy'

    # load data
    rgb_image = np.load(input_filepath)

    white_cell = rgb_image.min(axis=2)

    red_cell = np.clip(rgb_image[..., 0] - rgb_image[..., 1],
            a_min=rgb_image[..., 0].min(), a_max=rgb_image[..., 0].max())
    green_cell = np.clip(rgb_image[..., 1] - rgb_image[..., 2],
            a_min=rgb_image[..., 1].min(), a_max=rgb_image[..., 1].max())
    blue_cell = np.clip(rgb_image[..., 2] - rgb_image[..., 0],
            a_min=rgb_image[..., 2].min(), a_max=rgb_image[..., 2].max())

    # polarity: angular distribution of white
    tdist = white_cell.sum(axis=0)
    tdist /= tdist.sum() # normalize to total expression

    plt.figure(figsize=(3,1))
    plt.plot(tdist, 'k-')
    plt.xlabel(r'angle $\theta$')
    plt.ylabel('intensity')
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folderpath, 'demo_app_polarity.png'), dpi=600)
    plt.close()

    # correlation
    plt.figure(figsize=(3,2))
    for sat_cell, color in zip([red_cell, green_cell, blue_cell],
            ['red', 'green', 'blue']):
        response = corr_metric(white_cell, sat_cell)
        plt.plot(response, color=color, linestyle='solid')
    plt.xlabel('radius lag')
    plt.ylabel('correlation metric')
#    plt.xticks([])
    plt.yticks([])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folderpath, 'demo_app_correlation.png'), dpi=600)
    plt.close()
