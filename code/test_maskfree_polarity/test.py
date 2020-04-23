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
    # load data
    a = np.load('demo2_gs_rt.npy')

    cc = corr_metric(a, a)
    plt.plot(cc)
    plt.show()

