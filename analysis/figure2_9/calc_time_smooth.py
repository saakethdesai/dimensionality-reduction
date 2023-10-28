import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative

basis_path = "/qscratch/ashriva/Experiments/Code/dim_reduction/paper_submitted/results/"
num_latent_dim = 100

def curvature(x, y):
    
    t = np.arange(len(x))
    # Compute the first and second derivatives of x and y
    dx = derivative(lambda i: np.interp(i, t, x), t)
    dy = derivative(lambda i: np.interp(i, t, y), t)
    d2x = derivative(lambda i: np.interp(i, t, x), t, n=2)
    d2y = derivative(lambda i: np.interp(i, t, y), t, n=2)

    # Compute the curvature of the curve at each point
    return (dx*d2y - dy*d2x) / (dx**2 + dy**2)**(3/2) 

def timesmooth(ax, z, start):

    D = z.shape[1]
    print(start.shape, z.shape)

    for j in range(1):
        xx = z[start[j]:start[j+1], :] # 50
        # [N , D]
        xx_grad = np.gradient(xx, axis=0) # (50 ,D)
        xx_grad_sign = np.sign(xx_grad) #(50, D)

        orig_sign = xx_grad_sign[0, :] #(1, D)
        nder_change_sign = []
        for k in range(xx_grad_sign.shape[0]):
            new_sign = xx_grad_sign[k, :]
            count = np.count_nonzero(new_sign - orig_sign)
            nder_change_sign.append(count)

        ax.bar(np.arange(len(nder_change_sign)), nder_change_sign, color='r')

    return D

def constant_direction(ax, z, start, name=""):

    D = z.shape[1]

    for j in range(1):
        xx = z[start[j]:start[j+1], :] # 50
        # [N , D]
        xx_grad = np.gradient(xx, axis=0) # (50 ,D)
        xx_grad_sign = np.sign(xx_grad) #(50, D)

        nder_change_sign = []
        for k in range(xx_grad_sign.shape[0]-1):
            orig_sign = xx_grad_sign[k, :] #(1, D)
            new_sign = xx_grad_sign[k+1, :]
            count = np.count_nonzero(new_sign - orig_sign)
            nder_change_sign.append(count)

        ax.plot(np.arange(len(nder_change_sign)), nder_change_sign, label=name)

    return D

# Diffusion maps
z = np.load(basis_path+"spd_dmaps_z.npy")[:2800, :num_latent_dim]
start = np.arange(0,len(z),50)
fig, axs = plt.subplots(2,3, figsize=(15, 10))
D = constant_direction(axs[1,0], z, start)
axs[1,0].set_title("Diffusion maps", fontsize=16)
axs[1,0].set_xlabel("Time", fontsize=16)
axs[1,0].set_ylabel("Cumulative change in \n sign of gradients", fontsize=20)
axs[1,0].set_yticks([0.0, D], ['0', '$l_d$'])
# axs[1,0].plot(D*np.ones(50), 'k-')
axs[1,0].tick_params(axis='both', which='major', labelsize=15)
axs[1,0].tick_params(axis='both', which='minor', labelsize=8)

# Autoencoder
basis_path = "/qscratch/saadesa/dimensionality_reduction/dendrite/AE/new_dataset/100/"
z = np.load(basis_path+"z.npy")
start = np.where(z[:, -1] == 0)[0]
D = constant_direction(axs[1,1], z, start)

axs[1,1].set_title("Autoencoder", fontsize=16)
axs[1,1].set_xlabel("Time", fontsize=16)
axs[1,1].set_yticks([0.0, D], ['0', '$l_d$'])
# axs[1,1].plot(D*np.ones(50), 'k-')
axs[1,1].tick_params(axis='both', which='major', labelsize=15)
axs[1,1].tick_params(axis='both', which='minor', labelsize=8)

# PCA
basis_path = "/qscratch/saadesa/dimensionality_reduction/dendrite/PCA/new_dataset/100"
z = np.loadtxt(basis_path+"/transformed_data.txt")
start = np.arange(0,len(z),50)
D = constant_direction(axs[1,2], z, start)

axs[1,2].set_title("PCA", fontsize=16)
axs[1,2].set_xlabel("Time", fontsize=16)
axs[1,2].set_yticks([0.0, D], ['0', '$l_d$'])
# axs[1,2].plot(D*np.ones(50), 'k-')
axs[1,2].tick_params(axis='both', which='major', labelsize=15)
axs[1,2].tick_params(axis='both', which='minor', labelsize=8)

fig.tight_layout()
plt.savefig("smooth_time_latent.png")

plt.figure(figsize=(8, 6))

# for j in range(6, 9):
#     xx = z_red[start[j]:start[j+1], :10]
    
#     if (j-6 == 0):
#         plt.plot(xx[:, 0], 'r-')
#         #plt.bar(np.arange(len(nder_change_sign)), nder_change_sign, color='r')
#     elif (j-6 == 1):
#         plt.plot(xx[:, 0], 'b-')
#         #plt.bar(np.arange(len(nder_change_sign)), nder_change_sign, color='b')
#     elif (j-6 == 2):
#         plt.plot(xx[:, 0], 'g-')
#         #plt.bar(np.arange(len(nder_change_sign)), nder_change_sign, color='g')
        
# plt.xlabel("Time", fontsize=16)
# plt.ylabel("Latent dimension 1", fontsize=16)

# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)