import numpy as np
import h5py
import matplotlib.pyplot as plt


GT_DIR = "/qscratch/ashriva/Experiments/Data/final_dataset/"
Dmaps_DIR = "/qscratch/ashriva/Experiments/Code/dim_reduction/dmaps_jul_2023/results/latest/"
AE_DIR = "/qscratch/saadesa/dimensionality_reduction/results_for_unshuffled_data/pvd/AE/"
PCA_DIR = "/qscratch/saadesa/dimensionality_reduction/results_for_unshuffled_data/pvd/PCA/"

hdffile = GT_DIR+"pvd_unshuffled_data.h5"
hf = h5py.File(hdffile, 'r')
gt = hf['data'][:]
hf.close()
print("GT shape", gt.shape)

dmaps = np.load(Dmaps_DIR+"pvd_unshuffled_dmaps_1000.npy")
print("DMAPS shape", dmaps.shape)

ae = np.load(AE_DIR+"pvd_100_ae_recon_unshuffled.npy")
print("AE shape", ae.shape)

pca = np.load(PCA_DIR+"pvd_unshuffled500_pca.npy")
print("PCA shape", pca.shape)

no_samples = gt.shape[0]
time_steps = 50
start = np.arange(0,no_samples,time_steps)

GT_diff = []
AE_diff = []
DMAPS_diff = []
PCA_diff = []

AE_recon_err = []
DMAPS_recon_err  = []
PCA_recon_err  = []

cmap = "copper_r"
for n in [91]:
    for i, index in enumerate(range(start[n],start[n+1])):
        fig, axs = plt.subplots(2,2, figsize=(10, 8))
        axs = axs.reshape(-1)

        axs[0].imshow(gt[index],  cmap=cmap)
        axs[0].set_title("Ground truth", fontsize=20, color='k')

        axs[1].imshow(dmaps[index].reshape(256,256),  cmap=cmap)
        axs[1].set_title("Diffusion Maps", fontsize=20, color = '#ff7f00')

        axs[2].imshow(ae[index,0,0],  cmap=cmap)
        axs[2].set_title("Autoencoder", fontsize=20, color='#4575b4')

        axs[3].imshow(pca[index].reshape(256,256),  cmap=cmap)
        axs[3].set_title("PCA", fontsize=20, color='#d73027')

        GT_diff.append(np.linalg.norm(gt[index]-gt[index+1]))
        DMAPS_diff.append(np.linalg.norm(dmaps[index]-dmaps[index+1]))
        AE_diff.append(np.linalg.norm(ae[index]-ae[index+1]))
        PCA_diff.append(np.linalg.norm(pca[index]-pca[index+1]))

        AE_recon_err.append(np.linalg.norm(ae[index].reshape(-1) - gt[index].reshape(-1)))
        DMAPS_recon_err.append(np.linalg.norm(dmaps[index].reshape(-1) - gt[index].reshape(-1)))
        PCA_recon_err.append(np.linalg.norm(pca[index].reshape(-1) - gt[index].reshape(-1)))

        for ax, name in zip(axs, ["GT", "DMAPS", "AE", "PCA"]):
            ax.set_xticks(())
            ax.set_yticks(())
            ax.invert_yaxis()

            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(3)

        plt.tight_layout()
        plt.savefig("pvd_unshuffeled_recon/"+str(index)+".png", dpi=300,  bbox_inches = 'tight', pad_inches = .1)
        plt.close()

# print("GT time change")
# print(GT_diff)
# print("DMAPS time change")
# print(DMAPS_diff)
# print("AE time change")
# print(AE_diff)
# print("PCA time change")
# print(PCA_diff)

# print("AE error")
# print(AE_recon_err)
# print("DMAPS error")
# print(DMAPS_recon_err)
# print("PCA error")
# print(PCA_recon_err)