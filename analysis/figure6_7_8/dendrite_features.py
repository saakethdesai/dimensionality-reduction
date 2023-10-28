import os
import numpy as np
import h5py
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

cwd = os.getcwd()

RECON_DIR = "../../results/"
DATA_DIR = "/qscratch/ashriva/Experiments/Data/final_dataset/"

DATA_type = "dendrite_"
AE_LATENT_TYPE = 100
PCA_LATENT_TYPE = 100
DMAPS_LATENT_TYPE = 500
# AE_LATENT_TYPE = 100
# PCA_LATENT_TYPE = 100
# DMAPS_LATENT_TYPE = 100
hdffile = DATA_DIR+ "dendrite_512.h5"
image_size = 512 
image_ae = np.load(RECON_DIR + DATA_type + str(AE_LATENT_TYPE) + "_ae.npy")
image_pca = np.load(RECON_DIR + DATA_type + str(PCA_LATENT_TYPE) + "_pca.npy")
image_dmaps = np.load(RECON_DIR + DATA_type + str(DMAPS_LATENT_TYPE) + "_dmaps.npy")[:]

hf = h5py.File(hdffile, 'r')
image_gt = hf['test'][:]
hf.close()

print(image_gt.shape, image_ae.shape, image_pca.shape, image_dmaps.shape)

def scale(image):

    Max = np.max(image)
    Min = np.min(image)
    return (image-Min)/(Max - Min)

def calc_rdf(sample):

    samples = scale(sample)

    # sample[sample > 0.6] = 1
    # sample[sample < 0.4] = 0
    # sample[sample == 0.5] = 0.5
    sample = (sample-np.min(sample))/(np.max(sample)-np.min(sample))

    sample = sample.astype('float32')[0]
    
    I_x = sample[256, :].flatten()
    I_d = np.array([sample[i, i] for i in range(512)])
    
    x = np.arange(0, 512, 1)
    # print (x.shape, I_x.shape, I_d.shape)

    return x/image_size, I_x, I_d 

cmap = "copper_r"

# for index in [149]:
r1_gt = []
r2_gt = []
r1_ae = []
r2_ae = []
r1_pca = []
r2_pca = []
r1_dmaps = []
r2_dmaps = []
THRESHOLD = 0.75
for index in [100, 110, 120, 130, 140, 149]:

    fig = plt.figure(figsize=(20, 8))

    sub1 = plt.subplot(2, 4, 1)
    sub1.set_xticks(())
    sub1.set_yticks(())
    gt_plot = sub1.imshow(image_gt[index].reshape(image_size,image_size), cmap=cmap)
    sub1.invert_yaxis()
    sub1.grid(False)
    sub1.set_title("Ground truth", fontsize=20, color='k')
    for axis in ['top', 'bottom', 'left', 'right']:
        sub1.spines[axis].set_linewidth(3)
    divider = make_axes_locatable(sub1)
    cax = divider.append_axes('right', size="7%", pad=0.1)
    cbar = fig.colorbar(gt_plot, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    sub2 = plt.subplot(2, 4, 2)
    sub2.set_xticks(())
    sub2.set_yticks(())
    ae_plot = sub2.imshow(image_ae[index].reshape(image_size,image_size), cmap=cmap)
    sub2.invert_yaxis()
    sub2.grid(False)
    sub2.set_title("Autoencoder", fontsize=20, c='#4575b4')
    for axis in ['top', 'bottom', 'left', 'right']:
        sub2.spines[axis].set_linewidth(3)
    divider = make_axes_locatable(sub2)
    cax = divider.append_axes('right', size="7%", pad=0.1,)
    cbar = fig.colorbar(ae_plot, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    sub3 = plt.subplot(2, 4, 5)
    sub3.set_xticks(())
    sub3.set_yticks(())
    pca_plot = sub3.imshow(image_pca[index].reshape(image_size,image_size), cmap=cmap)
    sub3.invert_yaxis()
    sub3.grid(False)
    sub3.set_title("PCA", fontsize=20, c='#d73027')
    for axis in ['top', 'bottom', 'left', 'right']:
        sub3.spines[axis].set_linewidth(3)
    divider = make_axes_locatable(sub3)
    cax = divider.append_axes('right', size="7%", pad=0.1,)
    cbar = fig.colorbar(ae_plot, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    sub4 = plt.subplot(2, 4, 6)
    sub4.set_xticks(())
    sub4.set_yticks(())
    dmaps_plot = sub4.imshow(image_dmaps[index].reshape(image_size,image_size), cmap=cmap)
    sub4.invert_yaxis()
    sub4.grid(False)
    sub4.set_title("Dmaps", fontsize=20, c='#ff7f00')
    for axis in ['top', 'bottom', 'left', 'right']:
        sub4.spines[axis].set_linewidth(3)
    divider = make_axes_locatable(sub4)
    cax = divider.append_axes('right', size="7%", pad=0.1,)
    cbar = fig.colorbar(dmaps_plot, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    sub5 = plt.subplot(2, 4, (3,4))
    sub6 = plt.subplot(2, 4, (7,8))
    for axis in ['top', 'bottom', 'left', 'right']:
        sub5.spines[axis].set_linewidth(3)
        sub6.spines[axis].set_linewidth(3)

    r, rdf_gt_x, rdf_gt_d = calc_rdf(image_gt[index].reshape(1,image_size,image_size))
    r1_gt.append(np.sum(rdf_gt_x > THRESHOLD))
    r2_gt.append(np.sum(rdf_gt_d > THRESHOLD))
    print(index, " GT ", np.sum(rdf_gt_x > THRESHOLD), np.sum(rdf_gt_d > THRESHOLD))
    sub5.plot(r, rdf_gt_x, lw=5, label='GT', c='black', linestyle='solid')
    sub6.plot(r, rdf_gt_d, lw=5, label='GT', c='black', linestyle='solid')
    #sub5.scatter(r[peaks], rdf_gt[peaks], marker="*", color='black', zorder=1)

    r, rdf_ae_x, rdf_ae_d = calc_rdf(image_ae[index].reshape(1,image_size,image_size))
    r1_ae.append(np.sum(rdf_ae_x > THRESHOLD))
    r2_ae.append(np.sum(rdf_ae_d > THRESHOLD))
    print(index, " AE ", np.sum(rdf_ae_x > THRESHOLD), np.sum(rdf_ae_d > THRESHOLD))
    sub5.plot(r, rdf_ae_x, lw=5, label='AE', c='#3773b8', linestyle='solid')
    sub6.plot(r, rdf_ae_d, lw=5, label='AE', c='#3773b8', linestyle='solid')
    #sub5.scatter(r[peaks], rdf_ae[peaks], marker="*", color='black', zorder=1)

    r, rdf_pca_x, rdf_pca_d = calc_rdf(image_pca[index].reshape(1,image_size,image_size))
    sub5.plot(r, rdf_pca_x, lw=5, label='PCA', c='#e41a1c', linestyle='solid')
    sub6.plot(r, rdf_pca_d, lw=5, label='PCA', c='#e41a1c', linestyle='solid')
    #sub5.scatter(r[peaks], rdf_pca[peaks], marker="*", color='black', zorder=1)
    print(index, " PCA ", np.sum(rdf_pca_x > THRESHOLD), np.sum(rdf_pca_d > THRESHOLD))
    r1_pca.append(np.sum(rdf_pca_x > THRESHOLD))
    r2_pca.append(np.sum(rdf_pca_d > THRESHOLD))

    r, rdf_dmaps_x, rdf_dmaps_d = calc_rdf(image_dmaps[index].reshape(1,image_size,image_size))
    print(index, " Dmaps ", np.sum(rdf_dmaps_x > THRESHOLD), np.sum(rdf_dmaps_d > THRESHOLD))
    sub5.plot(r, rdf_dmaps_x, lw=5, label='Dmaps', c='#ff7f00', linestyle='solid')
    sub6.plot(r, rdf_dmaps_d, lw=5, label='Dmaps', c='#ff7f00', linestyle='solid')
    r1_dmaps.append(np.sum(rdf_dmaps_x > THRESHOLD))
    r2_dmaps.append(np.sum(rdf_dmaps_d > THRESHOLD))
    #sub5.scatter(r[peaks], rdf_dmaps[peaks], marker="*", color='black', zorder=1)

    sub5.set_ylabel("Intensity (a.u)", fontsize=23)
    sub6.set_ylabel("Intensity (a.u)", fontsize=23)
    sub5.set_xlabel("$r_0$", fontsize=23)
    sub6.set_xlabel("$r_1$", fontsize=23)
    #sub5.legend(fontsize=20)
    sub5.tick_params(axis='both', which='major', labelsize=18)
    sub5.tick_params(axis='both', which='minor', labelsize=12)
    # sub5.legend(fontsize=20, loc='upper right')
    sub6.tick_params(axis='both', which='major', labelsize=18)
    sub6.tick_params(axis='both', which='minor', labelsize=12)
    sub6.legend(fontsize=20, loc='upper right')
    sub5.set_position([sub5.get_position().x0+.03, sub5.get_position().y0+.05, 
                        sub5.get_position().width -.03, sub5.get_position().height-.05])
    sub6.set_position([sub6.get_position().x0+.03, sub6.get_position().y0+.05, 
                        sub6.get_position().width -.03, sub6.get_position().height-.05])

    plt.savefig("dendrite/rdf_tf"+str(index)+"_rdf.png",  dpi=300,  bbox_inches = 'tight', pad_inches = .1)
    plt.close()


# ---------------------------------------------------------------------------------------------

r1_gt = np.array(r1_gt)
r2_gt = np.array(r2_gt)
r1_ae = np.array(r1_ae)
r2_ae = np.array(r2_ae)
r1_pca = np.array(r1_pca)
r2_pca = np.array(r2_pca)
r1_dmaps = np.array(r1_dmaps)
r2_dmaps = np.array(r2_dmaps)

cwd = os.getcwd()

fig = plt.figure(figsize=(8, 6))
sub5 = plt.subplot(1, 1, 1)
for axis in ['top', 'bottom', 'left', 'right']:
    sub5.spines[axis].set_linewidth(3)

time = np.array([0, 10, 20, 30, 40, 50])
sub5.plot(time, abs(r1_ae-r1_gt)/r1_gt, lw=6, c='#3773b8', marker='o', markersize=12, linestyle='solid')
sub5.plot(time, abs(r2_ae-r2_gt)/r2_gt, lw=6, c='#3773b8', marker='o', markersize=12, linestyle='dashed')

sub5.plot(time, abs(r1_pca-r1_gt)/r1_gt, lw=4, c='#e41a1c', marker='o', markersize=12, linestyle='solid')
sub5.plot(time, abs(r2_pca-r2_gt)/r2_gt, lw=4, c='#e41a1c', marker='o', markersize=12, linestyle='dashed')

sub5.plot(time, abs(r1_dmaps-r1_gt)/r1_gt, lw=2, c='#ff7f00', marker='o', markersize=12, linestyle='solid')
sub5.plot(time, abs(r2_dmaps-r2_gt)/r2_gt, lw=2, c='#ff7f00', marker='o', markersize=12, linestyle='dashed')

# Adding legends
colors = ['#3773b8', '#e41a1c', '#ff7f00']
labels = ['Autoencoder', 'PCA', 'Diffusion Maps']
lw = [6, 4, 2]
for cc, col in enumerate(labels):
    sub5.plot(np.NaN, np.NaN, c=colors[cc], label=col, lw=lw[cc])

styles = ['-', '--']
ax2 = sub5.twinx()
for ss, sty in enumerate(styles):
    ax2.plot(np.NaN, np.NaN, ls=styles[ss],
             label=r'$r_{}$'.format(str(ss)), c='black')

ax2.get_yaxis().set_visible(False)

sub5.set_ylabel("relative absolute error", fontsize=25, labelpad=10)
sub5.set_xlabel("time", fontsize=25, labelpad=5)
#sub5.legend(fontsize=20)
sub5.tick_params(axis='both', which='major', labelsize=20)
sub5.tick_params(axis='both', which='minor', labelsize=10)
sub5.set_ylim(-.01, .2)
sub5.legend(loc=0, fontsize=20)
ax2.legend(loc=2, fontsize=20)

fig.tight_layout()
plt.savefig("r_vs_t_dendrite_error.png")

# --------------------------------------------------------------------------------

cwd = os.getcwd()

fig = plt.figure(figsize=(8, 6))
sub5 = plt.subplot(1, 1, 1)
for axis in ['top', 'bottom', 'left', 'right']:
    sub5.spines[axis].set_linewidth(3)

time = np.array([0, 10, 20, 30, 40, 50])
sub5.plot(time, r1_gt/512, lw=6, c='black', marker='o', markersize=12, linestyle='solid')
sub5.plot(time, r2_gt/512, lw=6, c='black', marker='o', markersize=12, linestyle='dashed')

sub5.plot(time, r1_ae/512, lw=2, c='#3773b8', marker='o', markersize=12, linestyle='solid')
sub5.plot(time, r2_ae/512, lw=2, c='#3773b8', marker='o', markersize=12, linestyle='dashed')

sub5.plot(time, r1_pca/512, lw=2, c='#e41a1c', marker='o', markersize=12, linestyle='solid')
sub5.plot(time, r2_pca/512, lw=2, c='#e41a1c', marker='o', markersize=12, linestyle='dashed')

#r, rdf_dmaps_x, rdf_dmaps_d = calc_rdf(image_dmaps[index].reshape(1,image_size,image_size))
sub5.plot(time, r1_dmaps/512, lw=2, c='#ff7f00', marker='o', markersize=12, linestyle='solid')
sub5.plot(time, r2_dmaps/512, lw=2, c='#ff7f00', marker='o', markersize=12, linestyle='dashed')

# Adding legends
colors = ['black', '#3773b8', '#e41a1c', '#ff7f00']
labels = ['GT', 'Autoencoder', 'PCA', 'Diffusion Maps']
lw = [6, 2, 2, 2]
for cc, col in enumerate(labels):
    sub5.plot(np.NaN, np.NaN, c=colors[cc], label=col, lw=lw[cc])

styles = ['-', '--']
ax2 = sub5.twinx()
for ss, sty in enumerate(styles):
    ax2.plot(np.NaN, np.NaN, ls=styles[ss],
             label=r'$r_{}$'.format(str(ss)), c='black')

ax2.get_yaxis().set_visible(False)

sub5.set_ylabel("relative absolute error", fontsize=25, labelpad=10)
sub5.set_xlabel("time", fontsize=25, labelpad=5)
#sub5.legend(fontsize=20)
sub5.tick_params(axis='both', which='major', labelsize=20)
sub5.tick_params(axis='both', which='minor', labelsize=10)

sub5.legend(loc=0, fontsize=20)
ax2.legend(loc=2, fontsize=20)

fig.tight_layout()
plt.savefig("r_vs_t_dendrite.png")
