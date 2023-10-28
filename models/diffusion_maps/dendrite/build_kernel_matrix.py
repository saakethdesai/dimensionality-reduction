"""
Load data for dmaps, get diffusion matrix
"""

import numpy as np
import h5py
import sys

VAR_DIR = "vars"

def store_data_in_hdffile(name_, data, hf, start, end):
    if (name_ not in hf):
        hf.create_dataset(name_, (np.append(Total_no_samples, Total_no_samples)),
                          'float64')

    hf[name_][start:end] = data

# ------------------------------ DMAPS --------------------
def get_diffusion_kernel(diffusions, epsilon):

    diffusions = np.exp(-diffusions / (epsilon))
    scales = np.sum(diffusions, axis=0)**.5
    diffusions = diffusions / (scales[:, None] * scales[None, :])

    np.save(VAR_DIR+"/dmaps_scales", scales)
    np.save(VAR_DIR+"/dmaps_normalized", diffusions)

    with h5py.File(VAR_DIR+"/dmaps_normalized.h5", 'w') as hf:
        hf.create_dataset("distances", data=diffusions, dtype='float64')

    return diffusions, scales


if __name__ == '__main__':

    with h5py.File(VAR_DIR+'/dmaps_kernel.h5', 'r') as hf:
        diffusions = hf['distances'][:]

    if(sys.argv[1] == "opt"):
        # Finding optimum epsilon
        epsilon = float(sys.argv[2])
        print("Searching: building kernel using epsilon", epsilon, flush=True)

    elif(sys.argv[1] == "train"):
        # Using the optimum epsilon
        epsilon = np.load(VAR_DIR+"/opt_epsilon.npy")
        print("Training: building kernel using epsilon", epsilon, flush=True)

    diffusions, scales = get_diffusion_kernel(diffusions, epsilon)
    # Note: eigenvectors of transition matrix are the same for any power kappa



