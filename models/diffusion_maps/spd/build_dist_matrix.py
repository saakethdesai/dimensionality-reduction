"""
Load data for dmaps, get distances matrix, in parallel
"""

import numpy as np
import h5py
from mpi4py import MPI
import time
import argparse

VAR_DIR = "vars"


parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
parser.add_argument('-mat', default=0, type=int)
args = parser.parse_args()

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

            local_H = hf[data_name][start:end]
    comm.Barrier()

local_kernel = np.zeros([local_no_samples, 0])
for r in range(num_procs):
    Hr = comm.bcast(local_H, root=r)
    distances = np.array([np.sum(np.abs(Hr - a)**2, axis=1) for a in local_H])
    local_kernel = np.append(local_kernel, distances, axis=1)

def store_data_in_hdffile(name_, data, hf, start, end):
    if (name_ not in hf):
        hf.create_dataset(name_, (np.append(Total_no_samples, Total_no_samples)),
                          'float64')

    hf[name_][start:end] = data

for r in range(num_procs):
    if(rank == r):
        with h5py.File(VAR_DIR+'/dmaps_kernel.h5', 'a') as hf:
            store_data_in_hdffile(
                "distances", local_kernel, hf, start, end)
    comm.Barrier()
t1 = time.time()

if(rank == 0):
    print("\t Time to build: ", t1 - t0)