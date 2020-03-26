import numpy as np
import matplotlib.pyplot as plt

import celltk

if __name__ == '__main__':
    w = 30
    x, y = np.meshgrid(range(w), range(w), indexing='xy')
    xc, yc = np.median(x), np.median(y)
    r = np.sqrt((x-xc)**2 + (y-yc)**2)
    t = np.arctan2(x-xc, y-yc)

    fig, axes = plt.subplots(ncols=2, nrows=2)
    cm = dict(cmap='coolwarm')
    axes[0, 0].imshow(x, **cm)
    axes[0, 0].set_title('X')
    axes[0, 1].imshow(y, **cm)
    axes[0, 1].set_title('Y')
    axes[1, 0].imshow(r, **cm)
    axes[1, 0].set_title('R (radius)')
    axes[1, 1].imshow(t, **cm)
    axes[1, 1].set_title(r'$\theta$ (angle)')

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle('Illustration of xy-coordinate and polar coordinate')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('demo_polar.png')
    plt.show()
