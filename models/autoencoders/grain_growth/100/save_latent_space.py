import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from glob import glob
import h5py

import os
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

torch.manual_seed(0)
np.random.seed(0)

class AE(nn.Module):
    def __init__(self, latent_dim):
        super(AE, self).__init__()
        
        self.econv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1) #16x256x256
        self.epool1 = nn.MaxPool2d(2) #16x128x128
        self.econv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1) #8x128x128
        self.epool2 = nn.MaxPool2d(2) #8x64x64
        self.econv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) #4x32x32
        self.epool3 = nn.MaxPool2d(2) #4x32x32
       
        self.elinear0 = nn.Linear(32*32*32, latent_dim)
        self.dlinear0 = nn.Linear(latent_dim, 32*32*32) 
        
        self.dconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2) #4x32x32
        self.dconv2 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2) #8x64x64
        self.dconv3 = nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2) #16x128x128
        
    def encode(self, x):
        x = self.econv1(x)
        x = F.leaky_relu(x)
        x = self.epool1(x)
        x = self.econv2(x)
        x = F.leaky_relu(x)
        x = self.epool2(x)
        x = self.econv3(x)
        x = F.leaky_relu(x)
        x = self.epool3(x)
        x = x.view((-1, 32*32*32))
        z = self.elinear0(x)
        return z 

    def decode(self, z): 
        z = self.dlinear0(z)
        z = F.leaky_relu(z)
        x = z.view(-1, 32, 32, 32) 
        x = self.dconv1(x)
        x = F.leaky_relu(x)
        x = self.dconv2(x)
        x = F.leaky_relu(x)
        x = self.dconv3(x)
        x = F.sigmoid(x)
        return x 
	
    def forward(self, x):
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x


def ae_loss(recon_x, x):
    mse = F.mse_loss(recon_x, x)
    return mse 


#----------------------------------------------#
batch_size = 1

cwd = os.getcwd()

hdffile = "/qscratch/saadesa/dimensionality_reduction/final_dataset/graingrowth_256.h5"
hf = h5py.File(hdffile, 'r')

train_data = hf['train'][:]
val_data = hf['val'][:956] 
test_data = hf['test'][:] 

data = np.vstack((train_data, test_data))
print (train_data.shape, test_data.shape, data.shape)

#create model
latent_dim = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ae = AE(latent_dim).to(device=device)
num_params = sum(p.numel() for p in ae.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)

cwd = os.getcwd()

os.chdir(cwd)
ae.load_state_dict(torch.load("ae_at_epoch10.pth", map_location=torch.device('cpu')))

z_array = []

for batch_idx, sample in enumerate(data):
    print (batch_idx)
    sample = sample.reshape((1, 1, 256, 256)) 
    sample = sample.astype('float32')
    sample = torch.from_numpy(sample)
    sample = sample.to(device)
    # ae reconstruction
    z = ae.encode(sample).detach().numpy().flatten()
    z_array.append(z)

z_array = np.array(z_array)
print (z_array.shape)
np.savetxt("z.txt", z_array)
