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

DATA_DIR = "/qscratch/ashriva/Experiments/Data/final_dataset/new_data/"
VAR_DIR = "vars"
RESULT_DIR = "Results"
hdffile = DATA_DIR+"spd_unshuffled_data.h5"

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

def encoder(X, args, opt_m, Number_of_samples):

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

    with h5py.File(VAR_DIR+'/unshuffeled_dmaps_kernel.h5', 'r') as hf:
        K_new = hf['distances'][:Number_of_samples]
    P_new = (K_new.T/np.sum(K_new, axis=1)).T
    del K_new

    plom_basis = np.load(VAR_DIR+"/dmaps_basis.npy")[:, :opt_m]
    plom_eigvals = np.flip(np.load(VAR_DIR+"/dmaps_eigvals.npy"))
    dmaps_test = np.matmul(np.matmul(P_new, plom_basis), np.diag(
        1/plom_eigvals[:opt_m]))

    np.save(VAR_DIR+"/latent_unshuffuled", dmaps_test)

def decoder(args,opt_m):

    phi = np.load(VAR_DIR+"/dmaps_basis.npy")[:, :opt_m]
    phi = np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T)

    if(args.mat == 0):
        data_name = "scaled_data"
    else:
        data_name = "pca_proj"

    with h5py.File(VAR_DIR+'/data.h5', 'r') as hf:
        X_train = hf[data_name][:]

    beta = np.dot(phi, X_train)
    np.save(VAR_DIR+"/Beta", beta)
    del X_train, phi

    # DMAPS Reconstruction
    phi_test = np.load(VAR_DIR+"/latent_unshuffuled.npy")[:,:opt_m]
    y_reconst = np.dot(phi_test, beta)

    del phi_test

    if(args.mat != 0):
        pca_eigvecs_inv = np.load(VAR_DIR+"/pca_scaled_eigvecs_inv.npy")
        pca_mean = np.load(VAR_DIR+"/pca_mean.npy")
        y_reconst = np.dot(y_reconst, pca_eigvecs_inv.T) + pca_mean
        del pca_eigvecs_inv, pca_mean

    scales = np.load(VAR_DIR+"/scaling_std.npy")
    centers = np.load(VAR_DIR+"/scaling_mean.npy")
    y_reconst = y_reconst * scales + centers
    return y_reconst

def get_test_data(hdffile, Number_of_samples):
    with h5py.File(hdffile, 'r') as hf:
        orig = hf['data'][:Number_of_samples]
        orig = orig.reshape(-1, 512*512)

    return orig, 1

if __name__ == '__main__':
    print("Test Error")

    parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
    parser.add_argument('-mat', default=0, type=int)
    parser.add_argument('-m', default=-1, type=int)
    args = parser.parse_args()

    Number_of_samples = 10000

    orig, n_iter = get_test_data(hdffile, Number_of_samples)
    reconCostDMaps = np.zeros(0)
    encoder(orig, args, args.m, Number_of_samples)
    y_reconst = decoder(args, args.m)

    np.save("/qscratch/ashriva/Experiments/Code/dim_reduction/results/latest/spd_unshuffled_dmaps_"+str(args.m), y_reconst)