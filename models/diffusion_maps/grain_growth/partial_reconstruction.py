import numpy as np
import sys
import matplotlib.pyplot as plt

import argparse
VAR_DIR = "vars"
RESULT_DIR = "Results"

def dmaps_decoder(args, g_part):
    Z = np.load(VAR_DIR+"/dmaps_projection.npy")

    # Inverse diffusion maps
    y_reconst = np.dot(g_part, Z.T)
    np.save(VAR_DIR+"/dmaps_reconstruction", y_reconst)

    del Z

    if(args.mat != 0):
        # Inverse pca
        pca_eigvecs_inv = np.load(VAR_DIR+"/pca_scaled_eigvecs_inv.npy")
        pca_mean = np.load(VAR_DIR+"/pca_mean.npy")
        y_reconst = np.dot(y_reconst, pca_eigvecs_inv.T) + pca_mean
        np.save(VAR_DIR+"/dmaps_pca_reconstruction", y_reconst)

        del pca_eigvecs_inv, pca_mean

    # Inverse scaling
    scales = np.load(VAR_DIR+"/scaling_std.npy")
    centers = np.load(VAR_DIR+"/scaling_mean.npy")
    y_reconst = y_reconst * scales + centers

    np.save(VAR_DIR+"/reconstruction", y_reconst)


def sample_projection(H, g):

    
    a = np.dot(g, np.linalg.inv(np.dot(np.transpose(g), g)))
    if H.shape[1] != a.shape[0]:
        H = H.T
    Z = np.dot(H, a)
    np.save(VAR_DIR+"/dmaps_projection", Z)


if __name__ == '__main__':

    print("Partial reconstruction", flush=True)
    parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
    parser.add_argument('-mat', default=0, type=int)
    args = parser.parse_args()

    g = np.load(VAR_DIR+"/dmaps_basis.npy")

    if(args.mat == 0): # Diffusion step
        H = np.load(VAR_DIR+"/scaled_data.npy")
    else:
        H = np.load(VAR_DIR+"/pca_projection.npy")

    error_arr = []
    eigvals = np.load(VAR_DIR+"/dmaps_eigvals.npy")
    N = len(eigvals)
    for num_components in range(N):

        sample_projection(H, g[:,:num_components])
        dmaps_decoder(args, g[:,:num_components])

        orig = np.load(VAR_DIR+"/train_data.npy")
        recon = np.load(VAR_DIR+"/reconstruction.npy")
        reconMatrixDMaps = recon
        reconCostDMaps = np.mean(np.power(reconMatrixDMaps - orig, 2), axis=1)
        mse = np.mean(reconCostDMaps.reshape(-1, 1))
        error_arr.append(mse)
        print("number of components %d reconstruction error: %f" % (num_components, mse))

        np.save(RESULT_DIR+"/error_rcon_v_num",error_arr)
    
    fig, ax = plt.subplots(1,1,figsize=[15,4])
    plt.plot(range(N), error_arr)
    plt.xlabel("number of dmap components")
    plt.ylabel("reconstruction train mse error")
    plt.yscale("log")
    plt.savefig(RESULT_DIR+"/train_recon_v_num.png")