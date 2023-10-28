import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from memory_profiler import profile
import cv2
import random
random.seed(a=1, version=2)

import argparse

DATA_DIR = "/qscratch/ashriva/Experiments/Data/final_dataset/"
VAR_DIR = "vars"

def get_data(num, frac=None):
    print("Read Grain growth data", flush=True)
    
    hdffile = DATA_DIR+"graingrowth_256.h5"
    with h5py.File(hdffile, 'r') as hf:
        X = hf['train'][:num]

    print("\Data Shape: ",X.shape, flush=True)
    np.save(VAR_DIR+"/train_data.npy", X)

def scaling():
    X = np.load(VAR_DIR+"/train_data.npy")
    print("Scaling data", flush=True)
    # Scaling --------------------------------------------------------
    means, scales = np.mean(X, axis=0), np.std(X, axis=0)
    # means, scales = np.zeros(X.shape[1]), np.zeros(X.shape[1])+1
    scales[scales == 0.0] = 1.0
    X = (X - means) / scales

    # Save scales and means
    np.save(VAR_DIR+"/scaling_mean", means)
    np.save(VAR_DIR+"/scaling_std",  scales)
    np.save(VAR_DIR+"/scaled_data",  X)
    with h5py.File(VAR_DIR+"/data.h5", 'w') as hf:
        hf.create_dataset("scaled_data", data=X, dtype='float64')

def pca_preprocess1():
    print("Performing PCA", flush=True)
    X = np.load(VAR_DIR+"/scaled_data.npy")

    N, n = X.shape
    means = np.mean(X, axis=0)
    X = X - means

    np.save(VAR_DIR+"/pca_meaned_data", X)
    np.save(VAR_DIR+"/pca_mean", means)
    del means

    cov = np.cov(X.T)
    print("\tPCA Covariance matrix size: %d x %d" % (cov.shape[0],cov.shape[1]))
    np.save(VAR_DIR+"/pca_cov", cov)
    with h5py.File(VAR_DIR+"/pca_cov.h5", 'w') as hf:
        hf.create_dataset("cov", data=cov, dtype='float64')
    del cov

def pca_preprocess2():
    print("Performing PCA on Transposed data", flush=True)
    X = np.load(VAR_DIR+"/scaled_data.npy")

    N, n = X.shape
    means = np.mean(X, axis=0)
    X = X - means

    np.save(VAR_DIR+"/pca_meaned_data", X)
    np.save(VAR_DIR+"/pca_mean", means)
    del means

    cov = np.dot(X, X.transpose())/(N-1)
    np.save(VAR_DIR+"/pca_cov", cov)
    print("\tPCA Covariance matrix size: %d x %d" % (cov.shape[0],cov.shape[1]))
    with h5py.File(VAR_DIR+"/pca_cov.h5", 'w') as hf:
        hf.create_dataset("cov", data=cov, dtype='float64')
    del cov

if __name__ == '__main__':
    """
    Read data, scale and then perform pca
    """

    parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
    parser.add_argument('-pca', default=0, type=int)
    args = parser.parse_args()

    N = 6720 # number of samples
    get_data(N)

    # Scaling train data
    scaling()

    if args.pca == 1:
        pca_preprocess1()
    elif args.pca == 2:
        pca_preprocess2()
    else:
        print("\tNo PCA performed")