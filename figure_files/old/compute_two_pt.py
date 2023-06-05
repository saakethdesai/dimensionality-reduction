import os
import numpy as np
import h5py
from matplotlib import pyplot as plt
from pymks import PrimitiveBasis
from pymks.stats import autocorrelate
from scipy.signal import find_peaks
from scipy import ndimage

cwd = os.getcwd()

basis = PrimitiveBasis(n_states=3)

RECON_DIR = "/qscratch/ashriva/Experiments/Code/dim_reduction/results/"
DATA_DIR = "/qscratch/ashriva/Experiments/Data/final_dataset/"

DATA_type = "pvd_"
LATENT_TYPE = 100
hdffile = DATA_DIR+ "pvd_data.h5"
image_size = 256
image_ae = np.load(RECON_DIR + DATA_type + str(LATENT_TYPE) + "_ae.npy")
image_pca = np.load(RECON_DIR + DATA_type + str(LATENT_TYPE) + "_pca.npy")
image_dmaps = np.load(RECON_DIR + DATA_type + str(LATENT_TYPE) + "_dmaps.npy")[:1911]

hf = h5py.File(hdffile, 'r')
image_gt = hf['test'][:1911]
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
    correlation = correlation[..., 0]
    print (correlation.shape)
    
    #radial average
    correlation_sample = correlation[0]
    x, y = np.meshgrid(np.arange(correlation_sample.shape[0]), np.arange(correlation_sample.shape[1]))
    R = np.sqrt((x-correlation_sample.shape[0]/2)**2 + (y-correlation_sample.shape[1]/2)**2)
    print (R.shape, np.min(R), np.max(R))

    f = lambda r: correlation_sample[(R >= r - 0.5) & (R < r + 0.5)].mean()
    r = np.linspace(np.min(R), np.max(R), 1000)
    rdf = np.vectorize(f)(r)
    
    return r, rdf

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

for index in range(image_gt.shape[0]):

    if (index == 42): #good plot

        fig = plt.figure(figsize=(16, 8))

        sub1 = plt.subplot(2, 4, 1)
        sub1.set_xticks(())
        sub1.set_yticks(())
        sub1.imshow(image_gt[index].reshape(image_size,image_size), cmap=cmap)
        sub1.invert_yaxis()
        sub1.grid(False)
        sub1.set_title("Ground truth", fontsize=20)
        for axis in ['top', 'bottom', 'left', 'right']:
            sub1.spines[axis].set_linewidth(3)

        sub2 = plt.subplot(2, 4, 2)
        sub2.set_xticks(())
        sub2.set_yticks(())
        sub2.imshow(image_ae[index].reshape(image_size,image_size), cmap=cmap)
        sub2.invert_yaxis()
        sub2.grid(False)
        sub2.set_title("Autoencoder", fontsize=20, c='#377eb8')
        for axis in ['top', 'bottom', 'left', 'right']:
            sub2.spines[axis].set_linewidth(3)

        sub3 = plt.subplot(2, 4, 5)
        sub3.set_xticks(())
        sub3.set_yticks(())
        sub3.imshow(image_pca[index].reshape(image_size,image_size), cmap=cmap)
        sub3.invert_yaxis()
        sub3.grid(False)
        sub3.set_title("PCA", fontsize=20, c='#e41a1c')
        for axis in ['top', 'bottom', 'left', 'right']:
            sub3.spines[axis].set_linewidth(3)

        sub4 = plt.subplot(2, 4, 6)
        sub4.set_xticks(())
        sub4.set_yticks(())
        sub4.imshow(image_dmaps[index].reshape(image_size,image_size), cmap=cmap)
        sub4.invert_yaxis()
        sub4.grid(False)
        sub4.set_title("Dmaps", fontsize=20, c='#ff7f00')
        for axis in ['top', 'bottom', 'left', 'right']:
            sub4.spines[axis].set_linewidth(3)
        
        plt.savefig(cwd+"/"+DATA_type + str(LATENT_TYPE)+ "/rdf_tf"+str(index)+"_img.png")
        plt.close()

        fig = plt.figure(figsize=(8, 6))
        sub5 = plt.subplot(1, 1, 1)
        for axis in ['top', 'bottom', 'left', 'right']:
            sub5.spines[axis].set_linewidth(3)
        
        r, rdf_gt = calc_rdf(image_gt[index].reshape(1,image_size,image_size))
        sub5.plot(r, rdf_gt, lw=4, label='GT', c='black')
        peaks = feature_detect(rdf_gt, 1)
        #sub5.scatter(r[peaks], rdf_gt[peaks], marker="*", color='black', zorder=1)

        gt_valley = r[peaks][0]
        gt_peak = r[peaks][1]

        r, rdf_ae = calc_rdf(image_ae[index].reshape(1,image_size,image_size))
        sub5.plot(r, rdf_ae, lw=4, label='AE', c='#3773b8')
        peaks = feature_detect(rdf_ae, 2)
        #sub5.scatter(r[peaks], rdf_ae[peaks], marker="*", color='black', zorder=1)

        diff_ae_valley = abs(r[peaks][0] - gt_valley) / image_size
        diff_ae_peak = abs(r[peaks][1] - gt_peak) / image_size
        diff_ae_peak_arr.append(diff_ae_peak)
        diff_ae_valley_arr.append(diff_ae_valley)

        r, rdf_pca = calc_rdf(image_pca[index].reshape(1,image_size,image_size))
        sub5.plot(r, rdf_pca, lw=4, label='PCA', c='#e41a1c')
        peaks = feature_detect(rdf_pca, 3)
        #sub5.scatter(r[peaks], rdf_pca[peaks], marker="*", color='black', zorder=1)

        diff_pca_valley = abs(r[peaks][0] - gt_valley) / image_size
        diff_pca_peak = abs(r[peaks][1] - gt_peak) / image_size
        diff_pca_peak_arr.append(diff_pca_peak)
        diff_pca_valley_arr.append(diff_pca_valley)

        r, rdf_dmaps = calc_rdf(image_dmaps[index].reshape(1,image_size,image_size))
        sub5.plot(r, rdf_dmaps, lw=4, label='Dmaps', c='#ff7f00')
        peaks = feature_detect(rdf_dmaps, 4)
        #sub5.scatter(r[peaks], rdf_dmaps[peaks], marker="*", color='black', zorder=1)

        diff_dmaps_valley = abs(r[peaks][0] - gt_valley) / image_size
        diff_dmaps_peak = abs(r[peaks][1] - gt_peak) / image_size
        diff_dmaps_peak_arr.append(diff_dmaps_peak)
        diff_dmaps_valley_arr.append(diff_dmaps_valley)

        sub5.set_ylabel("$C(r) (a.u)$", fontsize=25, labelpad=10)
        sub5.set_xlabel("$r$ (pixels)", fontsize=25, labelpad=5)
        #sub5.legend(fontsize=20)
        sub5.tick_params(axis='both', which='major', labelsize=20)
        sub5.tick_params(axis='both', which='minor', labelsize=10)

        fig.tight_layout()
        directory = DATA_type + str(LATENT_TYPE)
        print(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        plt.savefig(cwd+"/"+DATA_type + str(LATENT_TYPE)+ "/rdf_tf"+str(index)+"_rdf.png")

        #rdf_gt_list.append(rdf_gt)
        #rdf_ae_list.append(rdf_ae)
        #rdf_dmaps_list.append(rdf_dmaps)
        #rdf_pca_list.append(rdf_pca)
        #rdf_r_list.append(r)

'''
ARR_DIR = "/qscratch/ashriva/Experiments/Code/dim_reduction/paper_dmaps/paper_arrays/RDF_arrays/"
np.save(ARR_DIR + DATA_type + str(LATENT_TYPE) + "_rdf_gt_list", rdf_gt_list)
np.save(ARR_DIR + DATA_type + str(LATENT_TYPE) + "_rdf_ae_list", rdf_ae_list)
np.save(ARR_DIR + DATA_type + str(LATENT_TYPE) + "_rdf_dmaps_list", rdf_dmaps_list)
np.save(ARR_DIR + DATA_type + str(LATENT_TYPE) + "_rdf_pca_list", rdf_pca_list)
np.save(ARR_DIR + DATA_type + str(LATENT_TYPE) + "_rdf_radius_list", rdf_r_list)

np.save(DATA_type + str(LATENT_TYPE) + "_diff_ae_peak_arr", diff_ae_peak_arr)
np.save(DATA_type + str(LATENT_TYPE) + "_diff_ae_valley_arr", diff_ae_valley_arr)
np.save(DATA_type + str(LATENT_TYPE) + "_diff_pca_peak_arr", diff_pca_peak_arr)
np.save(DATA_type + str(LATENT_TYPE) + "_diff_pca_valley_arr", diff_pca_valley_arr)
np.save(DATA_type + str(LATENT_TYPE) + "_diff_dmaps_peak_arr", diff_dmaps_peak_arr)
np.save(DATA_type + str(LATENT_TYPE) + "_diff_dmaps_valley_arr", diff_dmaps_valley_arr)
'''