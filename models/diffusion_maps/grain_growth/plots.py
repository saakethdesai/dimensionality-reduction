import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse

VAR_DIR = "vars"
RESULT_DIR = "Results"

def linearplot(values, xlabel, ylabel, title, name):
    fig, ax = plt.subplots(1,1)
    ax.plot(values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    plt.savefig(RESULT_DIR+"/"+name+".png")
    plt.close()

parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
parser.add_argument('-mat', default=0, type=int)
args = parser.parse_args()

if(args.mat != 0):
    eigvals = np.flip(np.load(VAR_DIR+"/pca_eigvals.npy"))
    linearplot(eigvals, "Eigen components", 
                            "Eigen values", 
                            "PCA Eigen values", 
                            "pca_eigenvals")


eigvals = np.flip(np.load(VAR_DIR+"/dmaps_eigvals.npy"))
linearplot(eigvals[1:], "Eigen components",
                        "Eigen values",
                        "Dmaps Eigen values",
                        "dmaps_eigenvals")

eps_v_m = np.load(VAR_DIR+"/eps_vs_m.npy")
fig, ax = plt.subplots(1,1)
ax.scatter(eps_v_m[:,0],eps_v_m[:,1])
ax.set_xlabel("Epsilon")
ax.set_ylabel("opt number of dim")
plt.savefig(RESULT_DIR+"/eps_v_m.png")
plt.close()
