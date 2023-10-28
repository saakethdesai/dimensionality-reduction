import matplotlib.pyplot as plt
import numpy as np
VAR_DIR = "vars"
RESULT_DIR = "Results/"

eigvals = np.load(VAR_DIR+"/dmaps_eigvals.npy")
N = len(eigvals)

error_arr = np.zeros(0)
for i in range(100):
    error_arr = np.append(error_arr , np.load(RESULT_DIR+str(i)+"_error_arr.npy"))

np.save(RESULT_DIR+"/train_error_v_m",error_arr)

fig, ax = plt.subplots(1,1,figsize=[15,4])
plt.plot(range(N), error_arr)
plt.xlabel("number of dmap components")
plt.ylabel("reconstruction train mse error")
plt.yscale("log")
plt.savefig(RESULT_DIR+"/train_recon_v_m.png")