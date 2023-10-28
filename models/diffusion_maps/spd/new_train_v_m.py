from mpi4py import MPI
import numpy as np
import sys
import matplotlib.pyplot as plt
import itertools

import argparse
VAR_DIR = "vars"
RESULT_DIR = "Results/"

comm = MPI.COMM_WORLD
num_procs = 100#comm.Get_size()
rank = comm.Get_rank()

def run():

    # print("Partial reconstruction encoder", flush=True)
    parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
    parser.add_argument('-mat', default=0, type=int)
    args = parser.parse_args()

    error_arr = []
    eigvals = np.load(VAR_DIR+"/dmaps_eigvals.npy")
    N = len(eigvals)
    del eigvals
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

    # start = num[rank]
    # end = num[rank+1]

    # Temp code ---------------------------------------
    start_arr = np.load("comp_arr.npy").astype('int')
    iters = np.load("iter_arr.npy").astype('int')
    b = np.array(range(num[rank+1]-start_arr[rank]))+(start_arr[rank]+1)
    start = start_arr[rank]+1
    end = b[-1]
    k = iters[rank]
    # -----------------

    print("rank %d, start %d, end %d, num of elements: %d allocated load: %f" % (rank, start, end, (end-start) ,(end**2-start**2)*.5))

    comm.Barrier()
    try:
        error_arr = np.load(RESULT_DIR+str(rank)+"_error_arr.npy")
    except:
        error_arr = np.zeros(0)

    # for k, num_components in enumerate(range(start, end)):
    for num_components in range(start, end):
        k = k+1
        g = np.load(VAR_DIR+"/dmaps_basis.npy")[:, :num_components]
        a = np.dot(g, np.linalg.inv(np.dot(np.transpose(g), g)))

        if(args.mat == 0): # Diffusion step
            H = np.load(VAR_DIR+"/scaled_data.npy")
        else:
            H = np.load(VAR_DIR+"/pca_projection.npy")
        if H.shape[1] != a.shape[0]:
            H = H.T
        Z = np.dot(H, a)
        np.save(VAR_DIR+"/parallel_dmaps_projection_"+str(rank)+"_"+str(k), Z)
        del H
        del a

        y_reconst = np.dot(g, Z.T)
        del Z
        del g

        scales = np.load(VAR_DIR+"/scaling_std.npy")
        centers = np.load(VAR_DIR+"/scaling_mean.npy")
        y_reconst = y_reconst * scales + centers
        np.save(VAR_DIR+"/parallel_reconstruction_"+str(rank), y_reconst)

        del centers
        del scales

        orig = np.load(VAR_DIR+"/train_data.npy")
        reconCostDMaps = np.mean(np.power(y_reconst - orig, 2), axis=1)
        mse = np.mean(reconCostDMaps.reshape(-1, 1))
        error_arr = np.append(error_arr, mse)
        print("Train Rank:%d Iter:%d number of components:%d reconstruction error: %f" % (rank, k, num_components, mse))
        del orig
        del y_reconst

        np.save(RESULT_DIR+str(rank)+"_error_arr", error_arr)
    comm.Barrier()

    error_arr = comm.gather(error_arr,root=0)
    if(rank == 0):
        error_arr = list(itertools.chain.from_iterable(error_arr))
        np.save(RESULT_DIR+"/train_error_v_m",error_arr)
    
        fig, ax = plt.subplots(1,1,figsize=[15,4])
        plt.plot(range(N), error_arr)
        plt.xlabel("number of dmap components")
        plt.ylabel("reconstruction train mse error")
        plt.yscale("log")
        plt.savefig(RESULT_DIR+"/train_recon_v_m.png")

if __name__ == '__main__':
    run()