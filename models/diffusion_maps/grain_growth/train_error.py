import sys
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from scipy import stats

from decimal import Decimal

DATA_DIR = "/qscratch/ashriva/Experiments/Data/final_dataset/"
VAR_DIR = "vars"
RESULT_DIR = "Results"

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

def encoder(args, opt_m):

    X = np.load(VAR_DIR+"/train_data.npy")

    # Scale
    means = np.load(VAR_DIR+"/scaling_mean.npy")
    scales = np.load(VAR_DIR+"/scaling_std.npy")
    X = (X - means) / scales
    np.save(VAR_DIR+"/check_scaled_train",X)
    del means, scales

    if(args.mat != 0):
        # PCA Projection
        means = np.load(VAR_DIR+"/pca_mean.npy")
        X = X - means
        eigvecs = np.load(VAR_DIR+"/pca_scaled_eigvecs.npy")
        X = np.dot(eigvecs.T, X.T).T
        del means, eigvecs

    # DMAPS reduction
    #g = np.load(VAR_DIR+"/dmaps_red_basis.npy")
    g = np.load(VAR_DIR+"/dmaps_basis.npy")
    g = g[:,:opt_m]
    a = np.dot(g, np.linalg.inv(np.dot(np.transpose(g), g)))
    if X.shape[1] != a.shape[0]:
        print("transpose")
        X = X.T
    print(X.shape, a.shape)
    Z = np.dot(X, a)
    np.save(VAR_DIR+"/train_dmaps_projection", Z)
    del a
    del X

def decoder(args, opt_m):
    # DMAPS Reconstruction
    Z = np.load(VAR_DIR+"/train_dmaps_projection.npy")
    #g = np.load(VAR_DIR+"/dmaps_red_basis.npy")
    g = np.load(VAR_DIR+"/dmaps_basis.npy")
    g = g[:,:opt_m]
    y_reconst = np.dot(g, Z.T)

    del g, Z

    if(args.mat != 0):
        pca_eigvecs_inv = np.load(VAR_DIR+"/pca_scaled_eigvecs_inv.npy")
        pca_mean = np.load(VAR_DIR+"/pca_mean.npy")
        y_reconst = np.dot(y_reconst, pca_eigvecs_inv.T) + pca_mean
        del pca_eigvecs_inv, pca_mean

    scales = np.load(VAR_DIR+"/scaling_std.npy")
    centers = np.load(VAR_DIR+"/scaling_mean.npy")
    y_reconst = y_reconst * scales + centers

    np.save(VAR_DIR+"/train_reconstruction", y_reconst)

if __name__ == '__main__':
    print("Train Error")

    parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
    parser.add_argument('-mat', default=0, type=int)
    parser.add_argument('-m', default=-1, type=int)
    args = parser.parse_args()

    if(args.m == -1):
        # use optimum m
        opt_m = np.load(VAR_DIR+"/dmaps_components.npy")
    else:
        # use user defined 
        opt_m = args.m

    print("\tUsing Latent dimension: %d"%opt_m)

    encoder(args, opt_m)
    decoder(args, opt_m)

    # Reconstruction error
    X = np.load(VAR_DIR+"/train_data.npy")
    y_reconst = np.load(VAR_DIR+"/train_reconstruction.npy")
    reconCostDMaps = np.mean(np.power(y_reconst - X, 2), axis=1)
    np.save(VAR_DIR+"/sample_wise_train_error",reconCostDMaps)

    mse = np.mean(reconCostDMaps.reshape(-1, 1))
    print(" Train reconstruction error, :", mse, flush=True)

    # Reconstruction error plot sample wise
    fig, ax = plt.subplots(1,1)
    ax.plot(reconCostDMaps)
    ax.set_xlabel("Sample id")
    ax.set_ylabel("MSE error")
    ax.set_yscale("log")
    plt.savefig(RESULT_DIR+"/train_sample_wise.png")
    plt.close()

    # Histogram plot sample wise
    #NBR_BINS = freedman_diaconis(reconCostDMaps,'bins')

    fig, ax = plt.subplots(1,1)
    ax.hist(reconCostDMaps, 100)
    ax.set_xlabel("Error")
    ax.set_ylabel("Density")
    plt.savefig(RESULT_DIR+"/train_hist.png")
    plt.close()

    # -------------- IMAGE DATASET -----------------------
    orig = X
    recon = y_reconst
    Nx = int(np.sqrt(orig.shape[1]))
    for i in range(100):
        fig, axs = plt.subplots(1,2, figsize=(15,5))
        axs[0].imshow(orig[i].reshape(Nx,Nx))
        axs[0].set_title("Orig")
        axs[1].imshow(recon[i].reshape(Nx,Nx))
        axs[1].set_title("recons : error: %.2E"%(Decimal(reconCostDMaps[i])))
        plt.savefig(RESULT_DIR+"/train/"+str(i)+".png")
        plt.close()
        print("Train %d recons : error: %.2E"%(i, Decimal(reconCostDMaps[i])))

