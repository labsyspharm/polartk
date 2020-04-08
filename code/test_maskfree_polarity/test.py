import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcolormap

from scipy import signal

if __name__ == '__main__':
    # load data
    a = np.load('demo2_gs_rt.npy')

    # loop and plot
    for i in range(1, a.shape[0]+1):
        # calculate normalized correlation coefficient
        cc = np.zeros(a.shape[0]-i+1)
        for j in range(a.shape[0]-i+1):
            pattern = a[0:i, :]
            image = a[j:j+i, :]
            cc[j] = np.corrcoef(pattern.flatten(), image.flatten())[0, 1]

        # offset for visual inspection
        cc += (a.shape[0]-i)*0.1

        # color for visual inspection
        c_index = int(i/a.shape[0]*256)
        c = mcolormap.coolwarm(c_index)

        plt.plot(cc, color=c, alpha=0.7)

    plt.xlabel('radius from cell centroid (pixels)')
    plt.ylabel('cross-correlation')
    plt.show()
