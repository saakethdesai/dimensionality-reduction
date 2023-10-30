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

data = np.vstack((train_data, test_data))
print (train_data.shape, test_data.shape, data.shape)

#read shuffled data to retrain
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

    z = pca_model.transform(data)
    
    print ("TRANSFORMED DATA = ", z.shape)
    recon = pca_model.inverse_transform(z)
    print ("RECON DATA = ", recon.shape)

    #save arrays
    np.save("dendrite_unshuffled"+str(ncomp)+"_pca", recon)
    np.save("dendrite_unshuffled"+str(ncomp)+"_pca_z", z)
    np.save("dendrite_unshuffled"+str(ncomp)+"_pca_eigenvec", pca_model.components_)
    np.save("dendrite_unshuffled"+str(ncomp)+"_pca_eigenval", pca_model.explained_variance_)

    #compute MSE
    mse = np.mean((data - recon)**2)
    print (mse)



