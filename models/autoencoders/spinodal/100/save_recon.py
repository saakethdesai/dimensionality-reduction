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
        
        self.econv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1) #8x512x512
        self.epool1 = nn.MaxPool2d(2) #8x256x256
        self.econv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1) #16x256x256
        self.epool2 = nn.MaxPool2d(2) #16x128x128
        self.econv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) #32x128x128
        self.epool3 = nn.MaxPool2d(2) #32x64x64
        self.econv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) #32x64x64
        self.epool4 = nn.MaxPool2d(2) #32x32x32
        self.econv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) #32x32x32
        self.epool5 = nn.MaxPool2d(2) #32x16x16
       
        self.elinear0 = nn.Linear(32*16*16, 1000)
        self.elinear1 = nn.Linear(1000, latent_dim)
        self.dlinear0 = nn.Linear(latent_dim, 1000)
        self.dlinear1 = nn.Linear(1000, 32*16*16)
        
        self.dconv1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2) #32x32x32
        self.dconv2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2) #32x64x64
        self.dconv3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2) #16x128x128
        self.dconv4 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2) #8x256x256
        self.dconv5 = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2) #4x512x512
        self.dconv6 = nn.ConvTranspose2d(4, 1, kernel_size=3, stride=1, padding=1) #1x512x512
        
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
        x = self.econv4(x)
        x = F.leaky_relu(x)
        x = self.epool4(x)
        x = self.econv5(x)
        x = F.leaky_relu(x)
        x = self.epool5(x)
        x = x.view((-1, 32*16*16))
        z = self.elinear0(x)
        z = F.leaky_relu(z)
        z = self.elinear1(z)
        return z 

    def decode(self, z): 
        z = self.dlinear0(z)
        z = F.leaky_relu(z)
        z = self.dlinear1(z)
        z = F.leaky_relu(z)
        x = z.view(-1, 32, 16, 16) 
        x = self.dconv1(x)
        x = F.leaky_relu(x)
        x = self.dconv2(x)
        x = F.leaky_relu(x)
        x = self.dconv3(x)
        x = F.leaky_relu(x)
        x = self.dconv4(x)
        x = F.leaky_relu(x)
        x = self.dconv5(x)
        x = F.leaky_relu(x)
        x = self.dconv6(x)
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


hdffile = "/qscratch/saadesa/dimensionality_reduction/final_dataset/spd_data.h5"
hf = h5py.File(hdffile, 'r')

print ("Reading data")
data = hf['test'][:] 
print ("Done reading data")
print (data.shape)

data = data[:]

#create model
latent_dim = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ae = AE(latent_dim).to(device=device)
num_params = sum(p.numel() for p in ae.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)

os.chdir(cwd)
ae.load_state_dict(torch.load("ae_at_epoch50.pth", map_location=torch.device('cpu')))

recon_array = []

for batch_idx, sample in enumerate(data):
    print (batch_idx)
    sample = sample.reshape((1, 1, sample.shape[0], sample.shape[1]))
    sample = sample.astype('float32')
    sample = torch.from_numpy(sample)
    sample = sample.to(device)
    # ae reconstruction
    sample_recon = ae(sample)
    recon_array.append(sample_recon.detach().numpy())

recon_array = np.array(recon_array)
print (recon_array.shape)
np.save("spd_100_ae_recon_test_shuffled", recon_array)

import matplotlib.pyplot as plt
plt.imshow(recon_array[0,0,0])
plt.savefig('check.png')