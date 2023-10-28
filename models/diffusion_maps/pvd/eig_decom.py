"""
Load covariance / kernel matrix perform eigen decompisition
"""
import sys
import numpy as np
from scipy import linalg
import time
import argparse

VAR_DIR = "vars"

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
    parser.add_argument('-mat', default=0, type=int)
    args = parser.parse_args()

    if(args.mat == 0): # Diffusion step
        print("Performing Eigen decomposition of diffusion kernel", flush=True)
        mat = np.load(VAR_DIR+"/dmaps_normalized.npy")
        name = "dmaps"

    else: # PCA STEP
        print("Performing Eigen decomposition of Covariance", flush=True)
        mat = np.load(VAR_DIR+"/pca_cov.npy")
        name = "pca"

    t0 = time.time()
    eigvals, eigvecs = linalg.eigh(mat)
    print("\tsequential Eigen solver execution time ",time.time()-t0, flush=True)

    #assert np.allclose(eigvecs @ np.diag(eigvals), mat @ eigvecs)

    del mat
    if(args.mat == 2):
        X = np.load(VAR_DIR+"/pca_meaned_data.npy")
        eigvecs = np.dot(X.transpose(), eigvecs)
        eigvecs = eigvecs/np.linalg.norm(eigvecs, axis=0)

    np.save(VAR_DIR+"/"+name+"_eigvals"+".npy", eigvals)
    np.save(VAR_DIR+"/"+name+"_eigvecs"+".npy", eigvecs)