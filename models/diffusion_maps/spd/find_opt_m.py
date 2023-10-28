import numpy as np
import argparse

VAR_DIR = "vars"
RESULT_DIR = "Results"

if __name__ == '__main__':

    error = np.load(RESULT_DIR+"/train_error_v_m.npy")
    
    indx = error < 1e-4
    if(np.sum(indx) > 0):
        m = error.shape[0] - error[indx].shape[0]
    else:
        m = error.shape[0]

    np.save(VAR_DIR+"/dmaps_components", (m+1))

    eigvals = np.load(VAR_DIR+"/dmaps_eigvals.npy")
    eigvals = np.flip(eigvals)

    basis = np.load(VAR_DIR+"/dmaps_basis.npy")

    s = 1
    e = m
    red_basis = basis[:, s:(m+1)]
    red_eigvals = eigvals[s:(m+1)]

    np.save(VAR_DIR+"/dmaps_red_basis", red_basis)
    np.save(VAR_DIR+"/dmaps_red_eigvals", np.flip(red_eigvals))