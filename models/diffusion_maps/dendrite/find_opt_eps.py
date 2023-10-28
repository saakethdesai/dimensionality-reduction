

# _get_dmaps_optimal_epsilon -> _get_dmaps_dim_from_epsilon -> _get_dmaps_basis, _get_dmaps_optimal_dimension

# _get_dmaps_optimal_epsilon

import numpy as np
import subprocess
import argparse

VAR_DIR = "vars"

def _get_dmaps_optimal_dimension(epsilon):
    dmaps_L=0.01

    eigvals = np.load(VAR_DIR+"/dmaps_eigvals.npy")
    eigvals = np.flip(eigvals)

    m = len(eigvals) - 1
    for a in range(2, len(eigvals)):
        r = eigvals[a] / eigvals[1]
        if r < dmaps_L:
            m = a - 1
            break

    # print("\teps: %.6f"%(epsilon), "\tm: ", m, flush=True)
    # print("\tmin eig: %.3f"%(min(eigvals)), 
    #       "\tmax eig: %.3f"%(max(eigvals)),"\n", flush=True)
    return m

def run_cmd(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()

def _get_dmaps_dim_from_epsilon(eps, args):

    # _get_dmaps_basis
    run_cmd("python -u build_kernel_matrix.py opt " +str(eps))

    if(args.parallel):
        run_cmd("mpirun --npernode "+str(args.npernode)+" --n "+str(args.n)+" python -u eig_decom_parallel.py -mat 0")
    else:
        run_cmd("python -u eig_decom.py -mat 0")


    m = _get_dmaps_optimal_dimension(eps)

    return m

def optimize_epsilon(args):
    epsilon_list = [0.1, 1, 2, 8, 16, 32, 64, 100, 1000, 10000]
    eps_for_m_target = [1, 10, 100, 1000, 10000]
    eps_vs_m = []

    m_target_list = [_get_dmaps_dim_from_epsilon(eps, args) 
    for eps in eps_for_m_target]

    m_target = min(m_target_list)
    eps_m_target = eps_for_m_target[np.argmin(m_target_list)]
    upper_bound = eps_m_target
    lower_bound = epsilon_list[0]
    #m_lower_bound = _get_dmaps_dim_from_epsilon(lower_bound, args)
    #m_upper_bound = _get_dmaps_dim_from_epsilon(upper_bound, args)
    #print("\tm target: %d, upper bound: %d, low bound: %d"%(m_target, m_lower_bound, m_upper_bound), flush=True)
    print("\teps target: %.3f, low bound: %.3f, upper bound: %.3f\n"%(eps_m_target, lower_bound, upper_bound), flush=True)
    for eps in epsilon_list[1:]:
        m = _get_dmaps_dim_from_epsilon(eps, args)
        eps_vs_m.append([eps, m])
        if m > m_target:
            lower_bound = eps
            #m_lower_bound = _get_dmaps_dim_from_epsilon(lower_bound, args)
            #print("\tm: %d, target: %d, upper bound: %d, low bound: %d"%(m, m_target, m_lower_bound, m_upper_bound), flush=True)
            print("\teps: %.3f, target: %.3f, low bound: %.3f, upper bound: %.3f\n"%(eps, eps_m_target, lower_bound, upper_bound), flush=True)

        else:
            upper_bound = eps
            #m_upper_bound = _get_dmaps_dim_from_epsilon(upper_bound, args)
            #print("\tm: %d, target: %d, upper bound: %d, low bound: %d"%(m, m_target, m_lower_bound, m_upper_bound), flush=True)
            print("\teps: %.3f, target: %.3f, low bound: %.3f, upper bound: %.3f\n"%(eps, eps_m_target, lower_bound, upper_bound), flush=True)

            break
    
    while upper_bound - lower_bound > 0.5:
        middle_bound = (lower_bound+upper_bound)/2
        m = _get_dmaps_dim_from_epsilon(middle_bound, args)
        eps_vs_m.append([middle_bound, m])
        if m > m_target:
            lower_bound = middle_bound
        else:
            upper_bound = middle_bound
        print("\tm target: %d, low bound: %.3f, upper bound: %.3f"%(m_target, lower_bound, upper_bound))

    m = _get_dmaps_dim_from_epsilon(lower_bound, args)
    while m > m_target:
        lower_bound += .01
        m = _get_dmaps_dim_from_epsilon(lower_bound, args)
        eps_vs_m.append([lower_bound, m])
        print("\tm target: %d, low bound: %.3f, upper bound: %.3f"%(m_target, lower_bound, upper_bound))

    epsilon = lower_bound
    eps_vs_m = np.unique(eps_vs_m, axis=0)

    np.save(VAR_DIR+"/opt_epsilon",epsilon)
    #np.save(VAR_DIR+"/m_target",m_target)
    np.save(VAR_DIR+"/eps_vs_m",eps_vs_m)

    print("optimum epsilon", epsilon)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
    parser.add_argument('-npernode','--npernode', default=1, type=int)
    parser.add_argument('-n','--n', default=1, type=int)
    parser.add_argument('-p','--parallel', action='store_true')
    args = parser.parse_args()

    print("STEP: Finding optimum epsilon",flush=True)
    optimize_epsilon(args)