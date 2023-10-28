import numpy as np
import h5py
from matplotlib import pyplot as plt
from scipy.stats import mstats

DATA_DIR = "/qscratch/ashriva/Experiments/Data/final_dataset/"
arr_dir = "../../results/"

def errbar_cal(err):

    ymin = np.min(err)
    ymax = np.max(err)
    ymean = np.mean(err)
    ystd = np.std(err)
    # yerror = np.stack((ystd, ystd))
    quantiles = mstats.mquantiles(err, axis=0)
    # yerror = np.array([quantiles[1]-quantiles[0], quantiles[2]-quantiles[1]])
    yerror = np.array(ymean-quantiles[0], quantiles[2]-ymean)

    return [ymean, yerror]

def scale_minus1to1(old_value):

    old_min = np.min(old_value)
    old_max = np.max(old_value)
    new_max = 1
    new_min = -1

    new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min

    return new_value

def calc_err(gt, recon):
    print("\t", i, hdffile)
    print(gt.shape, recon.shape)
    if(len(gt.shape) == 3):
        err = np.sqrt(np.mean((recon-gt)**2, axis=(1,2)))
        print(np.mean(err))
        return err
    else:
        err = np.sqrt(np.mean((recon-gt)**2, axis=1))
        print(np.mean(err))
        return err
    # csm = []
    # for n in range(gt.shape[0]):
    #     a = scale_minus1to1(gt[n,:].reshape(-1))
    #     b = scale_minus1to1(recon[n,:].reshape(-1))
    #     csm.append(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    # return np.array(csm)

fig, axs = plt.subplots(2,2, figsize=(8, 6))

# AE ------------------------------------------------------------------------------------------------
print("AE")
ld_ae = [10, 50, 100, 1000]

# mse_test_SPD_ae = [0.14,0.074,0.046]
# mse_test_dendrite_ae = [.008, .0054, .0065]
# mse_test_pvd_ae = [0.04,0.028,0.024]
# mse_test_gg_ae = [.0003, .0002, .0003]
mse_test_SPD_ae = []
mse_test_SPD_ae_bar = []
mse_test_dendrite_ae = []
mse_test_dendrite_ae_bar = []
mse_test_pvd_ae = []
mse_test_pvd_ae_bar = []
mse_test_gg_ae = []
mse_test_gg_ae_bar = []

for i in ld_ae:
    hdffile = DATA_DIR+"spd_data.h5"
    hf = h5py.File(hdffile, 'r')
    mse_test_sample_SPD_ae = calc_err(hf['test'][:2800].reshape(2800,512*512),
                                        np.load(arr_dir+"spd_"+str(i)+"_ae.npy").reshape(2800,512*512))
    hf.close()
    
    hdffile = DATA_DIR+"dendrite_512.h5"
    hf = h5py.File(hdffile, 'r')
    mse_test_sample_dendrite_ae = calc_err(hf['test'][:].reshape(1400,512*512),
                                        np.load(arr_dir+"dendrite_"+str(i)+"_ae.npy").reshape(1400,512*512))
    hf.close()

    hdffile = DATA_DIR+"pvd_data.h5"
    hf = h5py.File(hdffile, 'r')
    mse_test_sample_pvd_ae = calc_err(hf['test'][:].reshape(1022,256*256),
                                        np.load(arr_dir+"pvd_"+str(i)+"_ae.npy").reshape(1022,256*256))
    hf.close()

    hdffile = DATA_DIR+"graingrowth_256.h5"
    hf = h5py.File(hdffile, 'r')
    mse_test_sample_gg_ae = calc_err(hf['test'][:].reshape(1911,256*256),
                                        np.load(arr_dir+"graingrowth_"+str(i)+"_ae.npy").reshape(1911,256*256))
    hf.close()

    temp = errbar_cal(mse_test_sample_SPD_ae)
    mse_test_SPD_ae.append(temp[0])
    mse_test_SPD_ae_bar.append(temp[1])

    temp = errbar_cal(mse_test_sample_dendrite_ae)
    mse_test_dendrite_ae.append(temp[0])
    mse_test_dendrite_ae_bar.append(temp[1])

    temp = errbar_cal(mse_test_sample_pvd_ae)
    mse_test_pvd_ae.append(temp[0])
    mse_test_pvd_ae_bar.append(temp[1])

    temp = errbar_cal(mse_test_sample_gg_ae)
    mse_test_gg_ae.append(temp[0])
    mse_test_gg_ae_bar.append(temp[1])

axs[0,0].errorbar(ld_ae, mse_test_SPD_ae, np.array(mse_test_SPD_ae_bar).T, color='#4575b4', ecolor='#4575b4',
            marker="x",
            capsize=5, linewidth=3,
            elinewidth=2,
            markeredgewidth=2,
            label="AE",)
axs[0,1].errorbar(ld_ae, mse_test_pvd_ae, np.array(mse_test_pvd_ae_bar).T, color='#4575b4', ecolor='#4575b4',
            marker="x",
            capsize=5, linewidth=3,
            elinewidth=2,
            markeredgewidth=2, label="AE")
axs[1,0].errorbar(ld_ae, mse_test_dendrite_ae, np.array(mse_test_dendrite_ae_bar).T, color='#4575b4', ecolor='#4575b4',
            marker="x",
            capsize=5, linewidth=3,
            elinewidth=2,
            markeredgewidth=2, label="AE")
axs[1,1].errorbar(ld_ae, mse_test_gg_ae, np.array(mse_test_gg_ae_bar).T, color='#4575b4', ecolor='#4575b4',
            marker="x",
            capsize=5, linewidth=3,
            elinewidth=2,
            markeredgewidth=2, label="AE")

# # PCA -----------------------------------------------------------------------------------------------------------
print("PCA")
ld_pca = [10, 50, 100, 500, 1000, 2000, 5000]

# mse_test_SPD_pca = [0.21, 0.17, 0.14]
# mse_test_dendrite_pca = [0.0101, 0.003, .0017]
# mse_test_pvd_pca = [.066,.044, .036]
# mse_test_gg_pca = [.003,.0007, .0004]

mse_test_SPD_pca = []
mse_test_SPD_pca_bar = []
mse_test_dendrite_pca = []
mse_test_dendrite_pca_bar = []
mse_test_pvd_pca = []
mse_test_pvd_pca_bar = []
mse_test_gg_pca = []
mse_test_gg_pca_bar = []

for i in ld_pca:
    print(i)
    hdffile = DATA_DIR+"spd_data.h5"
    hf = h5py.File(hdffile, 'r')
    mse_test_sample_SPD_pca = calc_err(hf['test'][:2800].reshape(2800,512*512),
                                        np.load(arr_dir+"spd_"+str(i)+"_pca.npy").reshape(2800,512*512))
    hf.close()

    temp = errbar_cal(mse_test_sample_SPD_pca)
    mse_test_SPD_pca.append(temp[0])
    mse_test_SPD_pca_bar.append(temp[1])

    if(i <= 1000):
        hdffile = DATA_DIR+"dendrite_512.h5"
        hf = h5py.File(hdffile, 'r')
        mse_test_sample_dendrite_pca = calc_err(hf['test'][:].reshape(1400,512*512),
                                            np.load(arr_dir+"dendrite_"+str(i)+"_pca.npy").reshape(1400,512*512))
        hf.close()

        temp = errbar_cal(mse_test_sample_dendrite_pca)
        mse_test_dendrite_pca.append(temp[0])
        mse_test_dendrite_pca_bar.append(temp[1])

    if(i < 5000):
        hdffile = DATA_DIR+"pvd_data.h5"
        hf = h5py.File(hdffile, 'r')
        mse_test_sample_pvd_pca = calc_err(hf['test'][:].reshape(1022,256*256),
                                            np.load(arr_dir+"pvd_"+str(i)+"_pca.npy").reshape(1022,256*256))
        hf.close()

    temp = errbar_cal(mse_test_sample_pvd_pca)
    mse_test_pvd_pca.append(temp[0])
    mse_test_pvd_pca_bar.append(temp[1])

    if(i <= 1000):
        hdffile = DATA_DIR+"graingrowth_256.h5"
        hf = h5py.File(hdffile, 'r')
        mse_test_sample_gg_pca = calc_err(hf['test'][:].reshape(1911,256*256),
                                            np.load(arr_dir+"graingrowth_"+str(i)+"_pca.npy").reshape(1911,256*256))
        hf.close()

        temp = errbar_cal(mse_test_sample_gg_pca)
        mse_test_gg_pca.append(temp[0])
        mse_test_gg_pca_bar.append(temp[1])

axs[0,0].errorbar(ld_pca, mse_test_SPD_pca, np.array(mse_test_SPD_pca_bar).T, color='#d73027', ecolor='#d73027',
             marker="^",
            capsize=5, linewidth=3,
            elinewidth=2,
            markeredgewidth=2, label="PCA", ls='dotted')
axs[0,1].errorbar(ld_pca, mse_test_pvd_pca, np.array(mse_test_pvd_pca_bar).T, color='#d73027', ecolor='#d73027',
             marker="^",
            capsize=5, linewidth=3,
            elinewidth=2,
            markeredgewidth=2, label="PCA", ls='dotted')
axs[1,0].errorbar(ld_pca[:5], mse_test_dendrite_pca, np.array(mse_test_dendrite_pca_bar).T, color='#d73027', ecolor='#d73027',
             marker="^",
            capsize=5, linewidth=3,
            elinewidth=2,
            markeredgewidth=2, label="PCA", ls='dotted')
axs[1,1].errorbar(ld_pca[:5], mse_test_gg_pca, np.array(mse_test_gg_pca_bar).T, color='#d73027', ecolor='#d73027',
             marker="^",
            capsize=5, linewidth=3,
            elinewidth=2,
            markeredgewidth=2, label="PCA", ls='dotted')

# DMAPS ---------------------------------------------------------------------------------------------------------
print("DMAPS")
ld_dmaps = [10, 50, 100, 500, 1000, 2000, 3500, 5000]

# mse_test_SPD_dmaps =[0.21438921218520055, 0.15089024110772026, 0.07724080756072452]
# mse_test_dendrite_dmaps = [0.07660997642443418, 0.07374515136564178, 0.06963072868968975]
# mse_test_gg_dmaps = [0.013617871835445298, 0.0112278702811154, 0.009944077905216888]
# mse_test_pvd_dmaps = [0.3854592274480463, 0.3466798601789943, 0.3318574605455828]

mse_test_SPD_dmaps = []
mse_test_SPD_dmaps_bar = []
mse_test_dendrite_dmaps = []
mse_test_dendrite_dmaps_bar = []
mse_test_pvd_dmaps = []
mse_test_pvd_dmaps_bar = []
mse_test_gg_dmaps = []
mse_test_gg_dmaps_bar = []

for i in ld_dmaps:
    hdffile = DATA_DIR+"spd_data.h5"
    hf = h5py.File(hdffile, 'r')
    mse_test_sample_SPD_dmaps = calc_err(hf['test'][:2800].reshape(2800,512*512),
                                        np.load(arr_dir+"spd_"+str(i)+"_dmaps.npy")[:2800].reshape(2800,512*512))
    hf.close()
    
    hdffile = DATA_DIR+"dendrite_512.h5"
    hf = h5py.File(hdffile, 'r')
    mse_test_sample_dendrite_dmaps = calc_err(hf['test'][:].reshape(1400,512*512),
                                        np.load(arr_dir+"dendrite_"+str(i)+"_dmaps.npy").reshape(1400,512*512))
    hf.close()

    hdffile = DATA_DIR+"pvd_data.h5"
    hf = h5py.File(hdffile, 'r')
    mse_test_sample_pvd_dmaps = calc_err(hf['test'][:].reshape(1022,256*256),
                                        np.load(arr_dir+"pvd_"+str(i)+"_dmaps.npy").reshape(1022,256*256))
    hf.close()

    hdffile = DATA_DIR+"graingrowth_256.h5"
    hf = h5py.File(hdffile, 'r')
    mse_test_sample_gg_dmaps = calc_err(hf['test'][:].reshape(1911,256*256),
                                        np.load(arr_dir+"graingrowth_"+str(i)+"_dmaps.npy").reshape(1911,256*256))
    hf.close()

    temp = errbar_cal(mse_test_sample_SPD_dmaps)
    mse_test_SPD_dmaps.append(temp[0])
    mse_test_SPD_dmaps_bar.append(temp[1])

    temp = errbar_cal(mse_test_sample_dendrite_dmaps)
    mse_test_dendrite_dmaps.append(temp[0])
    mse_test_dendrite_dmaps_bar.append(temp[1])

    temp = errbar_cal(mse_test_sample_pvd_dmaps)
    mse_test_pvd_dmaps.append(temp[0])
    mse_test_pvd_dmaps_bar.append(temp[1])

    temp = errbar_cal(mse_test_sample_gg_dmaps)
    mse_test_gg_dmaps.append(temp[0])
    mse_test_gg_dmaps_bar.append(temp[1])

axs[0,0].errorbar(ld_dmaps, mse_test_SPD_dmaps, np.array(mse_test_SPD_dmaps_bar).T, color='#ff7f00', ecolor='#ff7f00',
                          marker="o",
            capsize=5, linewidth=3,
            elinewidth=2,
            markeredgewidth=2, label="Dmaps",ls='--')
axs[0,1].errorbar(ld_dmaps, mse_test_pvd_dmaps, np.array(mse_test_pvd_dmaps_bar).T, color='#ff7f00', ecolor='#ff7f00',
                          marker="o",
            capsize=5, linewidth=3,
            elinewidth=2,
            markeredgewidth=2, label="Dmaps",ls='--')
axs[1,0].errorbar(ld_dmaps, mse_test_dendrite_dmaps, np.array(mse_test_dendrite_dmaps_bar).T, color='#ff7f00', ecolor='#ff7f00',
                         marker="o",
            capsize=5, linewidth=3,
            elinewidth=2,
            markeredgewidth=2,  label="Dmaps",ls='--')
axs[1,1].errorbar(ld_dmaps, mse_test_gg_dmaps, np.array(mse_test_gg_dmaps_bar).T, color='#ff7f00', ecolor='#ff7f00',
                         marker="o",
            capsize=5, linewidth=3,
            elinewidth=2,
            markeredgewidth=2,  label="Dmaps",ls='--')

# ------------------------------------------------------------------------------------------------------------------

axs[0,0].set_title("Spinodal", fontsize=15)
axs[0,1].set_title("Physical Vapor Deposition", fontsize=15)
axs[1,0].set_title("Dendrite", fontsize=15)
axs[1,1].set_title("Grain growth", fontsize=15)

axs[0,0].set_ylim(1e-2,1)
axs[0,1].set_ylim(1e-2,1)
axs[1,0].set_ylim(1e-4,1)
axs[1,1].set_ylim(1e-4,1)

for i, ax in enumerate(axs.reshape(-1)):
    ax.set_yscale("log")
    ax.set_xscale("log")
    if(i == 0):
        ax.legend()
    # ax.set_xlabel("Latent dimension size", fontsize=18)
    # ax.set_ylabel("Reconstruction error", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=8)

fig.text(0.5, -0.04, 'Latent dimension size', ha='center', fontsize=18)
fig.text(-0.04, 0.5, 'Root mean squared error', va='center', rotation='vertical', fontsize=18)

plt.tight_layout()
plt.savefig("compare_mse_latest.png", format="png", dpi=300, bbox_inches="tight", transparent=False)