import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as sp
from scipy.misc import derivative
from scipy import interpolate
from sklearn.manifold import TSNE
import h5py

def tsne(ax, data, name):
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embedded_data = tsne.fit_transform(data)

    # Plot the embedded data
    ax.scatter(embedded_data[:, 0], embedded_data[:, 1])
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title(name)

def stats(val_array):
    return np.mean(val_array)

def scale(val_array):

    M = np.max(val_array)
    m = np.min(val_array)
    return (val_array-m)/(M-m)

def accelreration_measure(curve):
    # Compute the acceleration of the curve
    acceleration = np.gradient(np.gradient(scale(curve), axis=0), axis=0)

    # Compute the smoothness of the curve
    smoothness = np.mean(np.sqrt(np.sum(acceleration**2, axis=1)))

    print(f"The smoothness of the curve is {smoothness:.2f}")

    return smoothness

def slope_direct(ax, z, name, plot2=True, color="red"):

    # grad = np.gradient(z, axis=0)
    print(z.shape)
    t = np.arange(len(z))
    grad = np.zeros([len(z),0])
    for j in range(z.shape[1]):
        cs = interpolate.CubicSpline(t,  z[:,j])
        dx = derivative(cs, t)
        # dx = derivative(lambda i: np.interp(i, t, z[:,j]), t)
        
        grad = np.append(grad, dx.reshape(-1,1),
                         axis=1)
    
    a = np.diag(np.dot(grad[1:], grad[:-1].T))
    a = 1-np.diag(sp.distance.cdist(grad[1:], grad[:-1], 'cosine'))
    if(plot2):
        ax.plot(np.arange(len(a)), np.arccos(a), color="red", label=r'Angular change, $\theta(t)$', linewidth=5)
    else:
        ax.plot(np.arange(len(a)), np.arccos(a), color=color, label=name, linewidth=5)
    # ax.set_title(name, fontsize=20)
    return np.arccos(a)

def cusps(ax, z, name=""):

    D = z.shape[1]

    xx = z # [N , D]
    xx_grad = np.gradient(xx, axis=0) # (N ,D)
    xx_grad_sign = np.sign(xx_grad) #(N, D)
    nder_change_sign = []
    for k in range(xx_grad_sign.shape[0]-1):
        orig_sign = xx_grad_sign[k, :] #(1, D)
        new_sign = xx_grad_sign[k+1, :]
        count = np.count_nonzero(new_sign - orig_sign)
        nder_change_sign.append(count/D)

    ax.bar(np.arange(len(nder_change_sign)), nder_change_sign, color="blue", label="Number of changes in \n gradient sign, " + r'$N(t)$')
    # ax.set_title(name, fontsize=20)

    return nder_change_sign

DATA_DIR = "/qscratch/ashriva/Experiments/Data/final_dataset/"
basis_path = "/qscratch/ashriva/Experiments/Code/dim_reduction/paper_submitted/results/latent_traj/"
basis_path2 = "/qscratch/saadesa/dimensionality_reduction/results_for_unshuffled_data/spd/"

hdffile = DATA_DIR+"spd_unshuffled_data.h5"
hf = h5py.File(hdffile, 'r')
no_samples = 10000 #hf['data'].shape[0]

time_steps = 100
data_type = "spd"
start = np.arange(0,no_samples,time_steps)
AE_Ndim = 100
DMAPS_Ndim = 2000
PCA_NDim = 1000
corners_array_dmaps = []
angle_array_dmaps = []
acc_array_dmaps = []
corners_array_pca = []
angle_array_pca = []
acc_array_pca = []
corners_array_ae = []
angle_array_ae = []
acc_array_ae = []

for n in range(len(start)-1):
    fig = plt.figure(figsize=(30, 12))
    fig.subplots_adjust(wspace=.4, hspace=.3)

    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    axs = [ax1, ax2, ax3]
    axs_twin = [ax.twinx() for ax in axs[:3]]

    # --------------------------------------------------------------------------------
    # DMAPS
    z_dmaps = np.load(basis_path+"/spd_dmaps_z.npy")[:10000]
    print(n, start[n], start[n+1], z_dmaps.shape, no_samples)
    corners = cusps(axs[2], z_dmaps[start[n]:start[n+1], :DMAPS_Ndim], "dmaps")
    angle = slope_direct(axs_twin[2], z_dmaps[start[n]:start[n+1], :DMAPS_Ndim], "dmaps")
    acc = accelreration_measure(z_dmaps[start[n]:start[n+1], :DMAPS_Ndim])
    angle_array_dmaps.append(stats(angle))
    corners_array_dmaps.append(stats(corners))
    acc_array_dmaps.append(stats(acc))

    # PCA
    z_pca = np.load(basis_path2+"PCA/new/spd_unshuffled"+str(PCA_NDim)+"_pca_z.npy")
    print(n, start[n], start[n+1], z_pca.shape, no_samples)
    corners = cusps(axs[0], z_pca[start[n]:start[n+1], :PCA_NDim], "PCA")
    angle = slope_direct(axs_twin[0], z_pca[start[n]:start[n+1], :PCA_NDim], "PCA")
    acc = accelreration_measure(z_pca[start[n]:start[n+1], :PCA_NDim])
    angle_array_pca.append(stats(angle))
    corners_array_pca.append(stats(corners))
    acc_array_pca.append(stats(acc))

    # AE
    z_ae = np.loadtxt(basis_path2+"AE/new/z.txt")
    print(n, start[n], start[n+1], z_ae.shape, no_samples)
    corners = cusps(axs[1], z_ae[start[n]:start[n+1], :AE_Ndim], "AE")
    angle = slope_direct(axs_twin[1], z_ae[start[n]:start[n+1], :AE_Ndim], "AE")
    acc = accelreration_measure(z_ae[start[n]:start[n+1], :AE_Ndim])
    angle_array_ae.append(stats(angle))
    corners_array_ae.append(stats(corners))
    acc_array_ae.append(stats(acc))

    # -------------------------------------------------------------------------------------
    axs[0].set_title("PCA", color = '#d73027',fontsize=50)
    axs[1].set_title("Autoencoder", color='#4575b4', fontsize=50)
    axs[2].set_title("Diffusion maps", color='#ff7f00', fontsize=50)

    for i in range(3):
        if(i == 0):
            axs[i].set_ylabel(r"$C(t)$", fontsize=45,  color='blue')
            axs[i].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        else:
            axs[i].set_yticks([])

        for axis in ['top', 'bottom', 'left', 'right']:
                    axs[i].spines[axis].set_linewidth(3)
        axs[i].set_xlabel("Time", fontsize=45)
        axs[i].set_xticks([0, 20, 40, 60, 80, 100])
        axs[i].set_ylim([0.0, 1])
        axs[i].tick_params(axis='both', which='major', labelsize=35)
        axs[i].tick_params(axis='both', which='minor', labelsize=25)
        # axs[i].legend(fontsize = 25, loc="upper left")

        if(i==2):
            axs_twin[i].set_ylabel(r'$\alpha(t)$', fontsize=45, color='red')
            axs_twin[i].set_yticks([0, 1, 2, 3])
        else:
            axs_twin[i].set_yticks([])
        axs_twin[i].set_ylim([0.0, np.pi])
        axs_twin[i].tick_params(axis='both', which='major', labelsize=35)
        axs_twin[i].tick_params(axis='both', which='minor', labelsize=25)
        axs[i].tick_params(axis='y', colors='blue')
        axs_twin[i].tick_params(axis='y', colors='red')
        # axs_twin[i].legend(fontsize = 25, loc="upper left", bbox_to_anchor=(0, .75))

    fig.tight_layout()
    plt.savefig(data_type+"_time_smooth/"+str(n)+".png", dpi=300,  bbox_inches = 'tight', pad_inches = 0)
    plt.close()

    # -------------------------------------------------------------------------------------------------
    fig = plt.figure(figsize=(30, 30))
    # fig.subplots_adjust(wspace=.4, hspace=.3)

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    trajectory = scale(z_pca[start[n]:start[n+1], :])
    ax1.scatter( trajectory[:,0],  trajectory[:,1],  trajectory[:,2])

    trajectory = scale(z_ae[start[n]:start[n+1], :])
    ax2.scatter( trajectory[:,0],  trajectory[:,1],  trajectory[:,2])

    trajectory = scale(z_dmaps[start[n]:start[n+1], :])
    ax3.scatter( trajectory[:,0],  trajectory[:,1],  trajectory[:,2])
    fig.tight_layout()
    plt.savefig(data_type+"_time_smooth/traj_"+str(n)+".png", dpi=300,  bbox_inches = 'tight', pad_inches = 0)
    plt.close()

    # -------------------------------------------------------------------------------------------------
    fig, axs = plt.subplots(1,3, figsize=(15,5))
    tsne(axs[0], z_pca[start[n]:start[n+1], :], 'PCA')
    tsne(axs[1], z_ae[start[n]:start[n+1], :], 'AE')
    tsne(axs[2], z_dmaps[start[n]:start[n+1], :], 'Dmaps')
    plt.savefig(data_type+"_time_smooth/tsne_"+str(n)+".png", dpi=300,  bbox_inches = 'tight', pad_inches = 0)
    plt.close()

import pandas as pd
import seaborn as sns

#create wide-form data
df_count = pd.DataFrame({'PCA': np.array(corners_array_pca),
                   'Autoencoder':  np.array(corners_array_ae),
                   'Diffusion maps': np.array(corners_array_dmaps)})
df_angle = pd.DataFrame({'PCA': np.array(angle_array_pca),
                   'Autoencoder':  np.array(angle_array_ae),
                   'Diffusion maps': np.array(angle_array_dmaps)})

# Define a list of colors for each xtick label
colors = ['#d73027', '#4575b4', '#ff7f00']

fig, axs = plt.subplots(1, 2, figsize=(24, 8))
#view data
a = sns.boxplot(ax=axs[0], x='variable', y='value', data=pd.melt(df_count), palette=colors)
axs[0].set_ylim([0, 0.5])
axs[0].set_xlabel('', fontsize=0)
axs[0].set_ylabel(r'$<C(t)>$', fontsize=45, color='blue')
axs[0].tick_params(axis='both', which='major', labelsize=32)
axs[0].tick_params(axis='both', which='minor', labelsize=25)
axs[0].tick_params(axis='y', colors='blue')
for axis in ['top', 'bottom', 'left', 'right']:
    axs[0].spines[axis].set_linewidth(3)
# Get the xtick labels
xtick_labels = axs[0].get_xticklabels()

# Set the color for each xtick label
for i, label in enumerate(xtick_labels):
    label.set_color(colors[i % len(colors)])

b = sns.boxplot(ax=axs[1], x='variable', y='value', data=pd.melt(df_angle), palette=colors)
axs[1].set_ylim([0.0, np.pi/2])
axs[1].set_yticks([0, 0.5, 1, 1.5])
axs[1].set_xlabel('', fontsize=0)
axs[1].set_ylabel(r'$<\alpha(t)>$', fontsize=45, color='red')
axs[1].tick_params(axis='both', which='major', labelsize=32)
axs[1].tick_params(axis='both', which='minor', labelsize=25)
axs[1].tick_params(axis='y', colors='red')
for axis in ['top', 'bottom', 'left', 'right']:
    axs[1].spines[axis].set_linewidth(3)
# Get the xtick labels
xtick_labels = axs[1].get_xticklabels()
# Set the color for each xtick label
for i, label in enumerate(xtick_labels):
    label.set_color(colors[i % len(colors)])

fig.tight_layout()
plt.savefig(data_type+"_Avg_smooth_measureslatent.png", dpi=300,  bbox_inches = 'tight', pad_inches = 0)
plt.close()

# -------------------------------------------------------------------------
df_acc = pd.DataFrame({'PCA': np.array(acc_array_pca),
                   'AE':  np.array(acc_array_ae),
                   'DMAPS': np.array(acc_array_dmaps)})

fig, axs = plt.subplots(1, 1, figsize=(15, 5))
b = sns.boxplot(ax=axs, x='variable', y='value', data=pd.melt(df_acc)).set(
            xlabel='Models',
            ylabel='Average acceleration')
fig.tight_layout()
plt.savefig(data_type+"_Avg_acceleration_latent.png", dpi=300,  bbox_inches = 'tight', pad_inches = 0)
plt.close()