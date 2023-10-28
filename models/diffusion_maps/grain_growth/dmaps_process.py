import numpy as np
import argparse

VAR_DIR = "vars"


def post_process_dmaps():
    print("\tTruncating Diffusion basis",flush=True)
    kappa = 1
    dmaps_L=0.1

    eigvecs = np.load(VAR_DIR+"/dmaps_eigvecs.npy")
    eigvals = np.load(VAR_DIR+"/dmaps_eigvals.npy")
    scales = np.load(VAR_DIR+"/dmaps_scales.npy")

    basis_vectors = eigvecs / scales[:, None]
    basis = basis_vectors * eigvals[None, :]**kappa

    basis = np.flip(basis, axis=1)
    np.save(VAR_DIR+"/dmaps_basis", basis)

    # eigvals = np.flip(eigvals)
    # eigvecs = np.flip(eigvecs, axis=1)

    # m = len(eigvals) - 1
    # for a in range(2, len(eigvals)):
    #     r = eigvals[a] / eigvals[1]
    #     if r < dmaps_L:
    #         m = a - 1
    #         break

    # s = 1
    # e = m+1
    # red_basis = basis[:, s:e]
    # red_eigvals = eigvals[s:e]
    

    # np.save(VAR_DIR+"/dmaps_components", e)
    # np.save(VAR_DIR+"/dmaps_red_basis", red_basis)
    # np.save(VAR_DIR+"/dmaps_red_eigvals", np.flip(red_eigvals))

    # return red_basis

if __name__ == '__main__':

    print("Projecting data to Diffusion space", flush=True)
    post_process_dmaps()
