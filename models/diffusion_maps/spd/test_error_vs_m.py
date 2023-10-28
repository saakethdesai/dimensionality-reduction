from mpi4py import MPI
import numpy as np
import sys
import matplotlib.pyplot as plt
import itertools

import argparse
VAR_DIR = "vars"
RESULT_DIR = "Results"

comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()

def dmaps_decoder(args, g_part, rank):
    Z = np.load(VAR_DIR+"/parallel_dmaps_projection_"+str(rank)+".npy")

    # Inverse diffusion maps
    y_reconst = np.dot(g_part, Z.T)
    #np.save(VAR_DIR+"/parallel_dmaps_reconstruction_"+str(rank), y_reconst)

    del Z

    if(args.mat != 0):
        # Inverse pca
        pca_eigvecs_inv = np.load(VAR_DIR+"/pca_scaled_eigvecs_inv.npy")
        pca_mean = np.load(VAR_DIR+"/pca_mean.npy")
        y_reconst = np.dot(y_reconst, pca_eigvecs_inv.T) + pca_mean
        #np.save(VAR_DIR+"/parallel_dmaps_pca_reconstruction_"+str(rank), y_reconst)

        del pca_eigvecs_inv, pca_mean

    # Inverse scaling
    scales = np.load(VAR_DIR+"/scaling_std.npy")
    centers = np.load(VAR_DIR+"/scaling_mean.npy")
    y_reconst = y_reconst * scales + centers

    np.save(VAR_DIR+"/parallel_reconstruction_"+str(rank), y_reconst)


def sample_projection(H, g, rank):

    a = np.dot(g, np.linalg.inv(np.dot(np.transpose(g), g)))
    if H.shape[1] != a.shape[0]:
        H = H.T
    Z = np.dot(H, a)
    np.save(VAR_DIR+"/parallel_dmaps_projection_"+str(rank), Z)


if __name__ == '__main__':

    print("Partial reconstruction", flush=True)
    parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
    parser.add_argument('-mat', default=0, type=int)
    args = parser.parse_args()

    g = np.load(VAR_DIR+"/dmaps_basis.npy")

    H = np.load(VAR_DIR+"/test_data.npy")

    # Scale
    means = np.load(VAR_DIR+"/scaling_mean.npy")
    scales = np.load(VAR_DIR+"/scaling_std.npy")
    H = (H - means) / scales
    del means, scales

    if(args.mat != 0):
        # PCA Projection
        means = np.load(VAR_DIR+"/pca_mean.npy")
        H = H - means
        eigvecs = np.load(VAR_DIR+"/pca_scaled_eigvecs.npy")
        H = np.dot(eigvecs.T, H.T).T
        del means, eigvecs

    error_arr = []
    eigvals = np.load(VAR_DIR+"/dmaps_eigvals.npy")
    N = len(eigvals)
    n = num_procs
    
    total_load = .5*N**2
    num = [0]
    load = []

    for i in range(1, n+1):
        if(i == n):
            num.append(N)
        else:
            t = N**2/n + num[i-1]**2
            num.append(int(np.sqrt(t)))

        load.append((num[i]**2-num[i-1]**2)/2)

    # print("status: ", load, np.sum(load), total_load, total_load/n)
    start = num[rank]
    end = num[rank+1]
    print("rank %d, start %d, end %d, num of elements: %d allocated load: %f" % (rank, start, end, (end-start) ,(end**2-start**2)*.5))
    comm.Barrier()

    for k, num_components in enumerate(range(start, end)):

        basis = g[:,:(num_components+1)]
        sample_projection(H, basis, rank)
        dmaps_decoder(args, basis, rank)

        orig = np.load(VAR_DIR+"/test_data.npy")
        recon = np.load(VAR_DIR+"/parallel_reconstruction_"+str(rank)+".npy")
        reconMatrixDMaps = recon
        reconCostDMaps = np.mean(np.power(reconMatrixDMaps - orig, 2), axis=1)
        mse = np.mean(reconCostDMaps.reshape(-1, 1))
        error_arr.append(mse)
        print("Test Rank:%d Iter:%d number of components:%d reconstruction error: %f" % (rank, k, num_components, mse))

    error_arr = comm.gather(error_arr,root=0)
    if(rank == 0):
        error_arr = list(itertools.chain.from_iterable(error_arr))
        np.save(RESULT_DIR+"/test_error_v_m",error_arr)
    
        fig, ax = plt.subplots(1,1,figsize=[15,4])
        plt.plot(range(N), error_arr)
        plt.xlabel("number of dmap components")
        plt.ylabel("reconstruction test mse error")
        plt.yscale("log")
        plt.savefig(RESULT_DIR+"/test_recon_v_m.png")