"""
postprocess pca
"""
import sys
import numpy as np
import h5py

VAR_DIR = "vars"

def pca_postprocess(trunc=False):
    print("\tPCA postprocessing")

    eigvecs = np.load(VAR_DIR+"/pca_eigvecs.npy")
    eigvals = np.load(VAR_DIR+"/pca_eigvals.npy")

    N = len(eigvals)

    if(trunc):
        print("\t\tperforming truncation")

        cumulative_energy = (1-1e-7)
        tot_eigvals = np.sum(eigvals)
        for i in range(len(eigvals)+1):
            if np.sum(eigvals[0:i])/tot_eigvals > (1-cumulative_energy):
                eigvals_trunc = eigvals[i-1:]
                break

        num_dropped_features = N - len(eigvals_trunc)
        eigvecs = eigvecs[:, num_dropped_features:]
        sqrt_eigvals = np.sqrt(eigvals_trunc)

        np.save(VAR_DIR+"/pca_eigval_trunc", eigvals_trunc)
        np.save(VAR_DIR+"/pca_scaled_eigvecs_inv", eigvecs * sqrt_eigvals)

        eigvecs = eigvecs / sqrt_eigvals
        np.save(VAR_DIR+"/pca_scaled_eigvecs", eigvecs)
        print("\t\tFeatures dropped from: %d to %d "%(N, N-num_dropped_features), flush=True)
    else:
        print("\t\tNo truncation, just removing eigen vectors with 0 eigvals")
        indx = ~np.isclose(eigvals, np.zeros(eigvals.shape), rtol=1e-05, atol=1e-08, equal_nan=False)
        eigvals = eigvals[indx]
        eigvecs = eigvecs[:, indx]
        
        sqrt_eigvals = np.sqrt(eigvals)
        np.save(VAR_DIR+"/pca_eigval_trunc", eigvals)
        np.save(VAR_DIR+"/pca_scaled_eigvecs_inv", eigvecs * sqrt_eigvals)

        eigvecs = eigvecs / sqrt_eigvals
        np.save(VAR_DIR+"/pca_scaled_eigvecs", eigvecs)


def pca_projection():
    print("\tPCA Projection")
    
    X = np.load(VAR_DIR+"/pca_meaned_data.npy")
    eigvecs = np.load(VAR_DIR+"/pca_scaled_eigvecs.npy")
    X = np.dot(eigvecs.T, X.T).T

    print("\t\twhitened Data Shape: ", X.shape, flush=True)
    np.save(VAR_DIR+"/pca_projection", X)
    with h5py.File(VAR_DIR+"/data.h5", 'a') as hf:
        hf.create_dataset("pca_proj", data=X, dtype='float64')


if __name__ == '__main__':

    print("Performing PCA Whitening", flush=True)

    pca_postprocess(trunc=False)
    pca_projection()