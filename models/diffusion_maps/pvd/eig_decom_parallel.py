import sys
import slepc4py
slepc4py.init(sys.argv)

from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
import math
import h5py
# import matplotlib.pyplot as plt
import time
import argparse


VAR_DIR = "vars"
float_tol = 1e-10

comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()

def linearplot(values, xlabel, ylabel, title, name):
    fig, ax = plt.subplots(1,1)
    ax.plot(values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(name+".png")
    plt.close()

def Eigsolve(mat, rang, dim):

    start, end = rang[0], rang[1]
    N, D = dim[0], dim[1]

    A = PETSc.Mat().create(PETSc.COMM_WORLD)
    A.setSizes([N, D])
    A.setUp()

    A.setValues(range(start, end), range(D), mat.reshape(-1))
    A.assemble()

    # Solving matrix in parallel
    E = SLEPc.EPS()
    E.create()
    E.setOperators(A)
    E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    E.setDimensions(D, PETSc.DECIDE)
    # E.setTolerances(1e-11, 10000)
    E.setFromOptions()
    E.solve()

    Print = PETSc.Sys.Print

    Print()
    Print("******************************")
    Print("*** SLEPc Solution Results ***")
    Print("******************************")
    Print()

    its = E.getIterationNumber()
    Print("Number of iterations of the method: %d" % its)

    eps_type = E.getType()
    Print("Solution method: %s" % eps_type)

    nev, ncv, mpd = E.getDimensions()
    Print("Number of requested eigenvalues: %d" % nev)

    tol, maxit = E.getTolerances()
    Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

    nconv = E.getConverged()
    Print("Number of converged eigenpairs %d" % nconv)

    if(rank == 0):
        eigvals = []
        eigvecs = np.zeros([D, 0])

    del mat

    if nconv > 0:
        # Create the results vectors
        vr, wr = A.getVecs()
        vi, wi = A.getVecs()
        #
        Print()
        Print("        k          ||Ax-kx||/||kx|| ")
        Print("----------------- ------------------")

        for i in range(nconv):
            k = E.getEigenpair(i, vr, vi)

            eigvec = comm.gather(vr.getArray(), root=0)
            if(rank == 0):
                if(abs(k.real) < float_tol):
                    eigvals.append(0)
                else:
                    eigvals.append(k.real)
                eigvec = np.concatenate(eigvec, axis=0).reshape(-1, 1)
                eigvecs = np.append(eigvecs, eigvec, axis=1)

            error = E.computeError(i)
            if k.imag != 0.0:
                Print("%d %9f%+9f j %12g" % (i, k.real, k.imag, error))
                # pass
            else:
                Print("%d %12f      %12g" % (i, k.real, error))
                # pass
        Print()

    if(rank == 0):
        return eigvals, eigvecs
    else:
        return None, None

if __name__ == '__main__':
    # reading matrix in parallel
    parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
    parser.add_argument('-mat', default=0, type=int)
    args = parser.parse_args()

    if(args.mat == 0): # Diffusion step
        print("Performing Eigen decomposition of diffusion kernel", flush=True)
        hdffile = VAR_DIR+"/dmaps_normalized.h5"
        with h5py.File(hdffile, 'r') as hf:
            N, D = hf["distances"].shape
            size = math.ceil(N / num_procs)
            start = rank * size
            if(rank == num_procs - 1):
                end = N
            else:
                end = start + size

            mat = hf["distances"][start: end, :]
        name = "dmaps"

    else: # PCA STEP
        print("Performing Eigen decomposition of Covariance", flush=True)
        hdffile = VAR_DIR+"/pca_cov.h5"
        with h5py.File(hdffile, 'r') as hf:
            N, D = hf["cov"].shape
            size = math.ceil(N / num_procs)
            start = rank * size
            if(rank == num_procs - 1):
                end = N
            else:
                end = start + size

            mat = hf["cov"][start: end, :]
        name = "pca"

    t0 = time.time()
    eigvals, eigvecs =  Eigsolve(mat, [start, end], [N, D])
    print("Parallel Eigen solver execution time ", time.time() - t0, flush = True)

    del mat
    if(rank == 0):
        if(args.mat == 2):
            X = np.load(VAR_DIR+"/pca_meaned_data.npy")
            eigvecs = np.dot(X.transpose(), eigvecs)
            eigvecs = eigvecs/np.linalg.norm(eigvecs, axis=0)

        eigvals = np.flip(eigvals)
        eigvecs = np.flip(eigvecs, axis=1)
        np.save(VAR_DIR+"/"+name+"_eigvals"+".npy", eigvals)
        np.save(VAR_DIR+"/"+name+"_eigvecs"+".npy", eigvecs)