"""
Load data for dmaps, get distances matrix, in parallel
"""

import numpy as np
import h5py
from mpi4py import MPI
import time
import argparse

VAR_DIR = "vars"
DATA_DIR = "/qscratch/ashriva/Experiments/Data/final_dataset/"
hdffile = DATA_DIR+"dendrite_512.h5"

parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
parser.add_argument('-mat', default=0, type=int)
args = parser.parse_args()

def diffusion_kernel(X1, X2, eps=.01):
    """
    X1 : N X D
    X2 : 1 X D
    return : N X 1
    """

    euclidean_distance = np.sum((X1-X2)**2, axis=1)
    return np.exp(- euclidean_distance / eps)

if(args.mat == 0):
    data_name = "scaled_data"
else:
    data_name = "pca_proj"

comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()

if(rank == 0):
    print("Building Diffusion distance using %s"% (data_name), flush=True)

t0 = time.time()
plom_epsilon = np.load(VAR_DIR+"/opt_epsilon.npy")
means = np.load(VAR_DIR+"/scaling_mean.npy")
scales = np.load(VAR_DIR+"/scaling_std.npy")
for r in range(num_procs):
    if(r == rank):
        with h5py.File(VAR_DIR+'/data.h5', 'r') as hf:
            Total_no_samples = hf[data_name].shape[0]
            div = Total_no_samples // num_procs

            start = rank * div
            if(rank == (num_procs-1)):
                end = Total_no_samples
            else:
                end = start + div

            local_no_samples = end- start

            local_H = hf[data_name][start:end].reshape(-1, 512*512)
    comm.Barrier()

with h5py.File(hdffile, 'r') as hf:
    Hr = hf['test'][:].reshape(-1, 512*512)
    Hr = (Hr - means) / scales

Tr_samples = Hr.shape[0]
local_kernel = np.array(
    [diffusion_kernel(local_H, a, plom_epsilon) for a in Hr])

def store_data_in_hdffile(name_, data, hf, start, end):
    if (name_ not in hf):
        hf.create_dataset(name_, (np.append(Tr_samples, Total_no_samples)),
                          'float64')

    hf[name_][:,start:end] = data

for r in range(num_procs):
    if(rank == r):
        with h5py.File(VAR_DIR+'/test_dmaps_kernel.h5', 'a') as hf:
            store_data_in_hdffile(
                "distances", local_kernel, hf, start, end)
    comm.Barrier()
t1 = time.time()

if(rank == 0):
    print("\t Time to build: ", t1 - t0)