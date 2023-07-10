import os
import numpy as np

from glob import glob

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import train_test_split
import pickle as pkl
from matplotlib import pyplot as plt

import h5py

home_dir = os.getcwd()

hdffile = "/qscratch/saadesa/dimensionality_reduction/final_dataset/dendrite_512.h5"
hf = h5py.File(hdffile, 'r')

print ("Reading data")
#train
train_data = hf['train'][:] 
#test
test_data = hf['test'][:]

print ("Done reading data")

print (train_data.shape, test_data.shape)

transformed_data = []
global_mse = []


#ncomp loop
ncomp_list = [10, 50, 100, 500, 1000]
for ncomp in ncomp_list:
    pca_model = PCA(n_components=ncomp)
    pca_model.fit(train_data)

    train_z = pca_model.transform(train_data)
    test_z = pca_model.transform(test_data)
    
    print ("TRANSFORMED DATA = ", test_z.shape)
    test_recon = pca_model.inverse_transform(test_z)
    print ("RECON DATA = ", test_recon.shape)

    #save arrays
    np.save("dendrite_"+str(ncomp)+"_pca", test_recon)
    np.save("dendrite_"+str(ncomp)+"_pca_z", test_z)
    np.save("dendrite_"+str(ncomp)+"_pca_eigenvec", pca_model.components_)
    np.save("dendrite_"+str(ncomp)+"_pca_eigenval", pca_model.explained_variance_)

    #compute MSE
    mse = np.mean((test_data - test_recon)**2)
    print (mse)

    #save images
    counter = 0
    for i in range(test_data.shape[0]):
        if (counter < 100):
            data_gt = test_data[i].reshape((512, 512)) 
            data_recon = test_recon[i].reshape((512, 512))
            #plotting 
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(data_gt)
            ax2.imshow(data_recon)
            plt.savefig("test"+"_"+str(ncomp)+"_"+str(i)+".png")
            plt.close()
            counter += 1

