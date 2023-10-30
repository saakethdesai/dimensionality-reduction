import os
import numpy as np

from glob import glob

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import train_test_split
import pickle as pkl
from matplotlib import pyplot as plt

import h5py

home_dir = os.getcwd()

hdffile = "/qscratch/saadesa/dimensionality_reduction/final_dataset/pvd_unshuffled_data.h5"
hf = h5py.File(hdffile, 'r')

print ("Reading data")
data = hf['data'][:] 
print ("Done reading data")
data = data.reshape((data.shape[0], data.shape[1]*data.shape[2]))
print (data.shape)

transformed_data = []
global_mse = []


#read shuffled data to retrain
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

transformed_data = []
global_mse = []



#ncomp loop
ncomp_list = [10, 50, 100, 500, 1000, 2000, 3500]
for ncomp in ncomp_list:
    pca_model = PCA(n_components=ncomp)
    pca_model.fit(train_data)

    z = pca_model.transform(data)
    
    print ("TRANSFORMED DATA = ", z.shape)
    recon = pca_model.inverse_transform(z)
    print ("RECON DATA = ", recon.shape)

    #save arrays
    np.save("pvd_unshuffled"+str(ncomp)+"_pca", recon)
    np.save("pvd_unshuffled"+str(ncomp)+"_pca_z", z)
    np.save("pvd_unshuffled"+str(ncomp)+"_pca_eigenvec", pca_model.components_)
    np.save("pvd_unshuffled"+str(ncomp)+"_pca_eigenval", pca_model.explained_variance_)

    #compute MSE
    mse = np.mean((data - recon)**2)
    print (mse)
