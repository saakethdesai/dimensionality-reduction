import os
import numpy as np
import h5py
from matplotlib import pyplot as plt
from pymks import PrimitiveBasis
from pymks.stats import autocorrelate
from scipy.signal import find_peaks
from scipy import ndimage
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
warnings.filterwarnings("ignore")

basis = PrimitiveBasis(n_states=2)

DATA_DIR = "/qscratch/ashriva/Experiments/Data/final_dataset/new_data/"
basis_path = "../../results/"
basis_path2 = "/qscratch/saadesa/dimensionality_reduction/results_for_unshuffled_data/spd"

DATA_type = "spd"
AE_LATENT_TYPE = 100
PCA_LATENT_TYPE = 1000
DMAPS_LATENT_TYPE = 2000
hdffile = DATA_DIR+ "spd_unshuffled_data.h5"
image_size = 512

image_ae = np.load(basis_path2+"/AE/new/spd_"+str(AE_LATENT_TYPE)+"_ae_recon_unshuffled.npy")[:1000].reshape(-1, 512, 512)
image_pca = np.load(basis_path2+"/PCA/new/spd_unshuffled"+str(PCA_LATENT_TYPE)+"_pca.npy")[:1000].reshape(-1, 512, 512)
image_dmaps = np.load(basis_path+"/spd_unshuffled_dmaps_"+str(DMAPS_LATENT_TYPE)+".npy")[:1000].reshape(-1, 512, 512)

hf = h5py.File(hdffile, 'r')
image_gt = hf['data'][:1000]
# image_gt = hf['test'][:]
hf.close()
print(image_gt.shape, image_ae.shape, image_pca.shape, image_dmaps.shape)

def feature_detect(data, A):

    # p, _ = find_peaks(data)
    gf = ndimage.gaussian_filter1d(data, sigma=1, order=1, mode='wrap') 

    p = np.diff(np.sign(gf)).nonzero()[0] + 1

    f1 = np.argmin(data[p])
    f2 = np.argmax(data[p][1:]) + 1

    # fig1, ax1 = plt.subplots()
    # ax1.plot(gf)
    # fig1.savefig(str(A)+".png")
    # plt.close(fig1)

    return [p[f1], p[f2]]

def calc_rdf(sample):

    # sample[sample > 0.6] = 1
    # sample[sample < 0.4] = 0
    # sample[sample == 0.5] = 0.5
    sample = (sample-np.min(sample))/(np.max(sample)-np.min(sample))

    sample = sample.astype('float32')
    correlation = autocorrelate(sample, basis, periodic_axes=(0,1))
    correlation = correlation[..., 1]
    # print (correlation.shape)
    
    #radial average
    correlation_sample = correlation[0]
    x, y = np.meshgrid(np.arange(correlation_sample.shape[0]), np.arange(correlation_sample.shape[1]))
    R = np.sqrt((x-correlation_sample.shape[0]/2)**2 + (y-correlation_sample.shape[1]/2)**2)
    # print (R.shape, np.min(R), np.max(R))

    f = lambda r: correlation_sample[(R >= r - 0.5) & (R < r + 0.5)].mean()
    r = np.linspace(np.min(R), np.max(R), 1000)
    rdf = np.vectorize(f)(r)
    
    return r/image_size, rdf

cmap = "copper_r"

diff_ae_peak_arr = []
diff_ae_valley_arr = []
diff_pca_peak_arr = []
diff_pca_valley_arr = []
diff_dmaps_peak_arr = []
diff_dmaps_valley_arr = []

rdf_ae_list = []
rdf_pca_list = []
rdf_dmaps_list = []
rdf_gt_list = []
rdf_r_list = []

# for index in range(100):
for index in [99]:
    print(index)
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
    sub2.set_title("Autoencoder", fontsize=20, color='#4575b4')
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
    sub3.set_title("PCA", fontsize=20, color='#d73027')
    for axis in ['top', 'bottom', 'left', 'right']:
        sub3.spines[axis].set_linewidth(3)
    divider = make_axes_locatable(sub3)
    cax = divider.append_axes('right', size="7%", pad=0.1,)
    cbar = fig.colorbar(pca_plot, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    sub4 = plt.subplot(2, 4, 6)
    sub4.set_xticks(())
    sub4.set_yticks(())
    dmaps_plot = sub4.imshow(image_dmaps[index].reshape(image_size,image_size), cmap=cmap)
    sub4.invert_yaxis()
    sub4.grid(False)
    sub4.set_title("Diffusion Maps", fontsize=20, color = '#ff7f00')
    for axis in ['top', 'bottom', 'left', 'right']:
        sub4.spines[axis].set_linewidth(3)
    divider = make_axes_locatable(sub4)
    cax = divider.append_axes('right', size="7%", pad=0.1,)
    cbar = fig.colorbar(dmaps_plot, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    sub5 = plt.subplot(1, 2, 2)
    r, rdf_gt = calc_rdf(image_gt[index].reshape(1,image_size,image_size))
    sub5.plot(r, rdf_gt, color= 'k', label="Ground truth",linewidth=5)
    peaks = feature_detect(rdf_gt, 1)
    sub5.scatter(r[peaks], rdf_gt[peaks], marker="*",s=10,  color='black', zorder=1)
    print(index," GT ", r[peaks])

    gt_valley = r[peaks][0]
    gt_peak = r[peaks][1]

    r, rdf_ae = calc_rdf(image_ae[index].reshape(1,image_size,image_size))
    sub5.plot(r, rdf_ae, color= '#4575b4', label="Autoencoder",linewidth=5)
    peaks = feature_detect(rdf_ae, 2)
    sub5.scatter(r[peaks], rdf_ae[peaks], marker="*",s=10,  color='black', zorder=1)
    print(index," AE ", r[peaks])

    diff_ae_valley = abs(r[peaks][0] - gt_valley) / image_size
    diff_ae_peak = abs(r[peaks][1] - gt_peak) / image_size
    diff_ae_peak_arr.append(diff_ae_peak)
    diff_ae_valley_arr.append(diff_ae_valley)

    r, rdf_pca = calc_rdf(image_pca[index].reshape(1,image_size,image_size))
    sub5.plot(r, rdf_pca, color= '#d73027', label="PCA",linewidth=5)
    peaks = feature_detect(rdf_pca, 3)
    sub5.scatter(r[peaks], rdf_pca[peaks], marker="*",s=10,  color='black', zorder=1)
    print(index," PCA ", r[peaks])

    diff_pca_valley = abs(r[peaks][0] - gt_valley) / image_size
    diff_pca_peak = abs(r[peaks][1] - gt_peak) / image_size
    diff_pca_peak_arr.append(diff_pca_peak)
    diff_pca_valley_arr.append(diff_pca_valley)

    r, rdf_dmaps = calc_rdf(image_dmaps[index].reshape(1,image_size,image_size))
    sub5.plot(r, rdf_dmaps, color='#ff7f00', label="Diffusion Maps",linewidth=5)
    peaks = feature_detect(rdf_dmaps, 4)
    sub5.scatter(r[peaks], rdf_dmaps[peaks], marker="*",s=10, color='black', zorder=1)
    print(index, " Dmaps ", r[peaks])

    diff_dmaps_valley = abs(r[peaks][0] - gt_valley) / image_size
    diff_dmaps_peak = abs(r[peaks][1] - gt_peak) / image_size
    diff_dmaps_peak_arr.append(diff_dmaps_peak)
    diff_dmaps_valley_arr.append(diff_dmaps_valley)

    sub5.set_ylabel("Two point correlation", fontsize=23)
    sub5.set_xlabel("r", fontsize=23)
    sub5.legend(fontsize=20)
    sub5.tick_params(axis='both', which='major', labelsize=18)
    sub5.tick_params(axis='both', which='minor', labelsize=12)
    sub5.legend(fontsize=20, loc='upper right')
    for axis in ['top', 'bottom', 'left', 'right']:
       sub5.spines[axis].set_linewidth(3)
    sub5.set_position([sub5.get_position().x0+.03, sub5.get_position().y0, 
                        sub5.get_position().width -.03, sub5.get_position().height])
    # fig.tight_layout()
    directory = DATA_type
    # print(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    plt.savefig(DATA_type+ "/rdf_tf"+str(index)+".png",  dpi=300,  bbox_inches = 'tight', pad_inches = .1)

    rdf_gt_list.append(rdf_gt)
    rdf_ae_list.append(rdf_ae)
    rdf_dmaps_list.append(rdf_dmaps)
    rdf_pca_list.append(rdf_pca)
    rdf_r_list.append(r)
