import sys
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from scipy import stats
import h5py
import math
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from decimal import Decimal

DATA_DIR = "/qscratch/ashriva/Experiments/Data/final_dataset/"
VAR_DIR = "vars"
RESULT_DIR = "Results"
hdffile = DATA_DIR+"graingrowth_256.h5"

def freedman_diaconis(data, returnas="width"):
    """
    Use Freedman Diaconis rule to compute optimal histogram bin width. 
    ``returnas`` can be one of "width" or "bins", indicating whether
    the bin width or number of bins should be returned respectively. 


    Parameters
    ----------
    data: np.ndarray
        One-dimensional array.

    returnas: {"width", "bins"}
        If "width", return the estimated width for each histogram bin. 
        If "bins", return the number of bins suggested by rule.
    """
    data = np.asarray(data, dtype=np.float_)
    IQR  = stats.iqr(data, rng=(25, 75), scale="raw", nan_policy="omit")
    N    = data.size
    bw   = (2 * IQR) / np.power(N, 1/3)

    if bw!=0 :
        if returnas=="width":
            result = bw
        else:
            datmin, datmax = data.min(), data.max()
            datrng = datmax - datmin
            result = int((datrng / bw) + 1)
        return(result)
    else:
        return(1)

def encoder(X, args):
    print(X.shape)
    # Nystrom projection ------------------------------
    # Scale
    means = np.load(VAR_DIR+"/scaling_mean.npy")
    scales = np.load(VAR_DIR+"/scaling_std.npy")
    X = (X - means) / scales
    del means, scales

    if(args.mat != 0):
        # PCA Projection
        means = np.load(VAR_DIR+"/pca_mean.npy")
        X = X - means
        eigvecs = np.load(VAR_DIR+"/pca_scaled_eigvecs.npy")
        X = np.dot(eigvecs.T, X.T).T
        del means, eigvecs

    with h5py.File(VAR_DIR+'/test_dmaps_kernel.h5', 'r') as hf:
        K_new = hf['distances'][:]
    P_new = (K_new.T/np.sum(K_new, axis=1)).T
    del K_new

    plom_basis = np.load(VAR_DIR+"/dmaps_basis.npy")
    plom_eigvals = np.flip(np.load(VAR_DIR+"/dmaps_eigvals.npy"))
    dmaps_test = np.matmul(np.matmul(P_new, plom_basis), np.diag(
        1/plom_eigvals))

    print(dmaps_test.shape)

    np.save(VAR_DIR+"/latent_unshuffuled", dmaps_test)

def get_test_data(hdffile):
    with h5py.File(hdffile, 'r') as hf:
        orig = hf['test'][:]
        orig = orig.reshape(-1, 256*256)

    return orig, 1

if __name__ == '__main__':
    print("Test Error")

    parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
    parser.add_argument('-mat', default=0, type=int)
    parser.add_argument('-m', default=-1, type=int)
    args = parser.parse_args()

    orig, n_iter = get_test_data(hdffile)
    reconCostDMaps = np.zeros(0)
    encoder(orig, args)