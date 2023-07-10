import os
import numpy as np

from glob import glob

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import train_test_split
import pickle as pkl
from matplotlib import pyplot as plt

import h5py

home_dir = os.getcwd()

hdffile = "/qscratch/saadesa/dimensionality_reduction/final_dataset/pvd_data.h5"
hf = h5py.File(hdffile, 'r')

print ("Reading data")
#train
train_data = hf['train'][:] 
#test
test_data = hf['test'][:]

print ("Done reading data")

train_data = train_data.reshape((train_data.shape[0], train_data.shape[1]*train_data.shape[2]))
test_data = test_data.reshape((test_data.shape[0], test_data.shape[1]*test_data.shape[2]))
print (train_data.shape, test_data.shape)

pca_model = PCA(n_components=None)

pca_model.fit(train_data)

train_z = pca_model.transform(train_data)
test_z = pca_model.transform(test_data)

print (train_z.shape, test_z.shape)

pkl.dump(pca_model, open("pca_model.pkl", "wb"))

transformed_data = []
global_mse = []

tmp_data = pca_model.transform(test_data)
print ("TRANSFORMED DATA = ", tmp_data.shape)

tmp_data_recon = pca_model.inverse_transform(tmp_data)
print ("RECON DATA = ", tmp_data_recon.shape)

#compute MSE
mse = np.mean((test_data - tmp_data_recon)**2)
print (mse)

#save images
counter = 0
for i in range(test_data.shape[0]):
    if (counter < 100):
        data_gt = test_data[i].reshape((256, 256)) 
        data_recon = tmp_data_recon[i].reshape((256, 256))
        #plotting 
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(data_gt)
        ax2.imshow(data_recon)
        plt.savefig("test"+str(i)+".png")
        plt.close()
        counter += 1


#save z and recon
np.save("pvd_all_pca", tmp_data_recon)
np.save("pvd_all_pca_z", tmp_data)

#save eigenvectors and eigenvalues
np.save("pvd_all_pca_eigenvec", pca_model.components_)
np.save("pvd_all_pca_eigenval", pca_model.explained_variance_)
np.save("pvd_all_pca_mean", pca_model.mean_)

