import numpy as np
import matplotlib.pyplot as plt
import os

VAR_DIR = "vars"
RESULT_DIR = "Results"

path = RESULT_DIR+"/latent/" 
if not os.path.exists(path):
    os.makedirs(path)

basis = np.load(VAR_DIR+"/dmaps_basis.npy")

n = basis.shape[0]
for i in range(basis.shape[1]):
    fig = plt.figure()
    plt.scatter(range(1,n+1),basis[:,i])
    plt.title("Eigen vector: "+str(i))
    plt.savefig(path+"scatter_"+str(i)+".png")
    plt.close()


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('1st comp')
ax.set_ylabel('2nd comp')
ax.set_zlabel('3rd comp')

ax.scatter(basis[:,0], basis[:,1], basis[:,2])
plt.title(f'DMAPS basi vectors (1,2,3)')
plt.savefig(path+"/3d_latent.png")


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('2nd comp')
ax.set_ylabel('3rd comp')
ax.set_zlabel('4th comp')

ax.scatter(basis[:,1], basis[:,2], basis[:,3])
plt.title(f'DMAPS basi vectors (2,3,4)')
plt.savefig(path+"/3d_latent_2.png")


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('1st comp')
ax.set_ylabel('2nd comp')
ax.set_zlabel('3rd comp')

ax.scatter(basis[:100,0], basis[:100,1], basis[:100,2])
plt.title(f'DMAPS basi vectors (1,2,3)')
plt.savefig(path+"/3d_latent_100.png")


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('2nd comp')
ax.set_ylabel('3rd comp')
ax.set_zlabel('4th comp')

ax.scatter(basis[:100,1], basis[:100,2], basis[:100,3])
plt.title(f'DMAPS basi vectors (2,3,4)')
plt.savefig(path+"/3d_latent_2_100.png")