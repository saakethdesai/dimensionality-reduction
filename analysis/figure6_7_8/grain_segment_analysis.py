import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable

RECON_DIR = "../../results/"
DATA_DIR = "/qscratch/ashriva/Experiments/Data/final_dataset/"
AE_LATENT_TYPE = 100
PCA_LATENT_TYPE = 100
DMAPS_LATENT_TYPE = 500
# AE_LATENT_TYPE = 100
# PCA_LATENT_TYPE = 100
# DMAPS_LATENT_TYPE = 100
image_size = 256
cmap = "copper_r"

def verify(n, image_gt, centroids_gt, image, centroids_im, name):

    idx_tuple = sort_grains(centroids_gt, centroids_im)

    gt_idx = np.zeros(image_gt.shape) - 1
    for index in np.unique(image_gt):
        true_idx = image_gt == index
        for key, value in idx_tuple.items():
            if value[0] == index:
                gt_idx[true_idx] = key

    im_idx = np.zeros(image.shape) - 1
    for index in np.unique(image):
        true_idx = image == index
        for key, value in idx_tuple.items():
            if value[1] == index:
                im_idx[true_idx] = key

    # Define the discrete value boundaries
    M = max(np.unique(image_gt).shape[0],
            np.unique(image).shape[0])
    # Create a colormap with discrete colors
    # cmap = plt.cm.jet  # define the colormap
    # Get the 'tab20' colormap
    base_colormap = plt.get_cmap('tab20')

    # Define the number of colors needed
    num_colors = M+1

    # Create a new colormap with more colors
    cmap = plt.cm.colors.ListedColormap(
        base_colormap(np.linspace(0, 1, num_colors))
    )

    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist[0] = (.5, .5, .5, 1.0)

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.linspace(-1, M, num=(M+2))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig, axs = plt.subplots(1,2, figsize=(10,5))
    im = axs[0].imshow(gt_idx, cmap=cmap, norm=norm)
    for i, p in enumerate(centroids_gt):
        axs[0].annotate(str(i), xy=(p[0]+0.5, p[1]+0.5),
                    ha='center', va='center', color='black')
    fig.colorbar(im, ax=axs[0])

    im = axs[1].imshow(im_idx, cmap=cmap, norm=norm)
    for i, p in enumerate(centroids_im):
        axs[1].annotate(str(i), xy=(p[0]+0.5, p[1]+0.5),
                    ha='center', va='center', color='black')
    fig.colorbar(im, ax=axs[1])
    plt.savefig("gg/"+name+str(n)+".png")
    plt.close()

def scale(image):

    Max = np.max(image)
    Min = np.min(image)
    return (image-Min)/(Max - Min)

def grain_stats(model, microstrcture, threshold=.75):
    microstrcture = scale(microstrcture)
    sx, sy =  microstrcture.shape
    total_area = sx*sy
    microstrcture[microstrcture < threshold] = 0
    microstrcture[microstrcture > threshold] = 1
    seg_grains = cv2.connectedComponentsWithStats(microstrcture.astype(np.uint8), 
                                                  connectivity=4)
    num_grains = seg_grains[0] - 1
    print(model, ": number of grains detected ",num_grains)

    grain_radial_array = []
    centroids = seg_grains[3][1:]
    for i in range(1, num_grains+1):
        area = np.sum(seg_grains[1] == i)
        # Equivalent radius
        equivalent_radius = np.sqrt(area/np.pi)
        grain_radial_array.append(equivalent_radius/sx)

    return grain_radial_array, seg_grains[1], centroids

def sort_grains(gt_centroids,recon_centroids):

    # find closest indexes
    indexes_dict = {}
    k = 0 
    for i, val1 in enumerate(gt_centroids):
        min_diff = float('inf')
        for j, val2 in enumerate(recon_centroids):
            diff = np.linalg.norm(val1 - val2)
            if diff < min_diff:
                min_diff = diff
                indexes = [i, j]
        if indexes:
            # print(gt_centroids[indexes[0]], recon_centroids[indexes[1]])
            # indexes_array.append(indexes)
            indexes_dict[k] = indexes
            k = k + 1
    return indexes_dict

hdffile = h5py.File(DATA_DIR+"graingrowth_256.h5",'r')
image_gt = hdffile['test']
image_ae = np.load(RECON_DIR + "graingrowth_" + str(AE_LATENT_TYPE) + "_ae.npy")
image_pca = np.load(RECON_DIR + "graingrowth_" + str(PCA_LATENT_TYPE) + "_pca.npy")
image_dmaps = np.load(RECON_DIR + "graingrowth_" + str(DMAPS_LATENT_TYPE) + "_dmaps.npy")

dmaps_diff_size_array = []
pca_diff_size_array = []
ae_diff_size_array = []

for index in range(100):
    print(index)
    grain_size_gt, seg_image_gt, centroids_gt = grain_stats('gt', 
                                                  image_gt[index].reshape(256,256))
    grain_size_ae, seg_image_ae, centroids_ae = grain_stats('AE',
                                                  image_ae[index].reshape(256,256), .85)
    grain_size_pca, seg_image_pca, centroids_pca = grain_stats('PCA',
                                                    image_pca[index].reshape(256,256), .65)
    grain_size_dmaps, seg_image_dmaps, centroids_dmaps = grain_stats('dmaps',
                                                        image_dmaps[index].reshape(256,256), .75)
    
    # index_dict = sort_grains(centroids_gt, centroids_ae)
    verify(index, seg_image_gt, centroids_gt, seg_image_ae, centroids_ae, "AE")
    verify(index, seg_image_gt, centroids_gt, seg_image_pca, centroids_pca, "PCA")
    verify(index, seg_image_gt, centroids_gt, seg_image_dmaps, centroids_dmaps, "Dmaps")

    # error_dmaps = abs(np.mean(grain_size_gt)-np.mean(grain_size_dmaps))
    # error_pca = abs(np.mean(grain_size_gt)-np.mean(grain_size_pca))
    # error_ae = abs(np.mean(grain_size_gt)-np.mean(grain_size_ae))
    # dmaps_diff_size_array.append(error_dmaps)
    # pca_diff_size_array.append(error_pca)
    # ae_diff_size_array.append(error_ae)

    fig = plt.figure(figsize=(20, 8))

    sub1 = plt.subplot(2, 4, 1)
    sub1.set_xticks(())
    sub1.set_yticks(())
    # sub1.imshow(image_gt[index].reshape(image_size,image_size), cmap=cmap)
    gt_plot = sub1.imshow(image_gt[index].reshape(256,256),cmap='copper')
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
    # sub2.imshow(image_ae[index].reshape(image_size,image_size), cmap=cmap)
    ae_plot = sub2.imshow(image_ae[index].reshape(256,256), cmap='copper')
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
    # sub3.imshow(image_pca[index].reshape(image_size,image_size), cmap=cmap)
    pca_plot = sub3.imshow(image_pca[index].reshape(256,256), cmap='copper')
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
    # sub4.imshow(image_dmaps[index].reshape(image_size,image_size), cmap=cmap)
    dmaps_plot = sub4.imshow(image_dmaps[index].reshape(256,256),cmap='copper')
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
    Nbins = 50
    ## Create the histogram
    # hist, bin_edges = np.histogram(grain_size_gt, bins=Nbins)
    ## Calculate the midpoints of the bins
    # bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    # sub5.plot(bin_midpoints, hist, color= 'k', label="Ground truth")

    x = np.linspace(-.1, .4, 1000)

    data = grain_size_gt
    kde = gaussian_kde(data)
    # Compute the density values for the x-axis values
    density = kde(x)
    loc = np.argmax(density)
    print(index, " GT:", x[loc])
    # Plot the density curve
    sub5.plot(x, density, color= 'k', label="Ground truth",linewidth=5)

    # hist, bin_edges = np.histogram(grain_size_ae, bins=Nbins)
    # # Calculate the midpoints of the bins
    # bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    # sub5.plot(bin_midpoints, hist, color= '#4575b4', label="Autoencoder")
    data = grain_size_ae
    kde = gaussian_kde(data)
    # Compute the density values for the x-axis values
    density = kde(x)
    loc = np.argmax(density)
    print(index, " AE:", x[loc])
    # Plot the density curve
    sub5.plot(x, density, color= '#4575b4', label="Autoencoder",linewidth=5)

    # hist, bin_edges = np.histogram(grain_size_pca, bins=Nbins)
    # # Calculate the midpoints of the bins
    # bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    # sub5.plot(bin_midpoints, hist, color= '#d73027', label="PCA")
    data = grain_size_pca
    kde = gaussian_kde(data)
    # Compute the density values for the x-axis values
    density = kde(x)
    loc = np.argmax(density)
    print(index, " PCA:", x[loc])
    # Plot the density curve
    sub5.plot(x, density, color= '#d73027', label="PCA",linewidth=5)

    # hist, bin_edges = np.histogram(grain_size_dmaps, bins=Nbins)
    # # Calculate the midpoints of the bins
    # bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    # sub5.plot(bin_midpoints, hist, color= '#ff7f00', label="Diffusion Maps")
    data = grain_size_dmaps
    kde = gaussian_kde(data)
    # Compute the density values for the x-axis values
    density = kde(x)
    loc = np.argmax(density)
    print(index, " Dmaps:", x[loc])
    # Plot the density curve
    sub5.plot(x, density, color= '#ff7f00', label="Diffusion Maps",linewidth=5)
    sub5.tick_params(axis='both', which='major', labelsize=18)
    sub5.tick_params(axis='both', which='minor', labelsize=12)
    sub5.set_xlim(0,.3)
    sub5.set_xlabel("Effective normalized grain size",  fontsize=23)
    sub5.set_ylabel("Density",  fontsize=23)

    # sub5.plot(image_gt[index].reshape(image_size,image_size)[75], label="gt")
    # sub5.plot(image_pca[index].reshape(image_size,image_size)[75], label="pca")
    # sub5.plot(image_ae[index].reshape(image_size,image_size)[75], label="ae")
    # sub5.plot(image_dmaps[index].reshape(image_size,image_size)[75], label="dmaps")
    sub5.legend(fontsize=20, loc='upper right')
    for axis in ['top', 'bottom', 'left', 'right']:
       sub5.spines[axis].set_linewidth(3)

    plt.savefig("gg/"+str(index)+".png",  dpi=300,  bbox_inches = 'tight', pad_inches = .1)
    plt.close()