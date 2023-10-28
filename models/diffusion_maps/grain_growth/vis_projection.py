import numpy as np
import argparse
import matplotlib.pyplot as plt

RESULT_DIR = "Results"
VAR_DIR = "vars"
path = RESULT_DIR+"/latent/" 

def encoder(args):

    X = np.load(VAR_DIR+"/train_data.npy")

    # Scale
    means = np.load(VAR_DIR+"/scaling_mean.npy")
    scales = np.load(VAR_DIR+"/scaling_std.npy")
    X = (X - means) / scales
    np.save(VAR_DIR+"/check_scaled_train",X)
    del means, scales

    if(args.mat != 0):
        # PCA Projection
        means = np.load(VAR_DIR+"/pca_mean.npy")
        X = X - means
        eigvecs = np.load(VAR_DIR+"/pca_scaled_eigvecs.npy")
        X = np.dot(eigvecs.T, X.T).T
        del means, eigvecs

    # DMAPS reduction
    #g = np.load(VAR_DIR+"/dmaps_red_basis.npy")
    g = np.load(VAR_DIR+"/dmaps_basis.npy")
    a = np.dot(g, np.linalg.inv(np.dot(np.transpose(g), g)))
    if X.shape[1] != a.shape[0]:
        print("transpose")
        X = X.T
    print(X.shape, a.shape)
    Z = np.dot(X, a)

    return Z.T

parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
parser.add_argument('-mat', default=0, type=int)
parser.add_argument('-m', default=-1, type=int)
args = parser.parse_args()

Z = encoder(args)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('1st comp')
ax.set_ylabel('2nd comp')
ax.set_zlabel('3rd comp')

ax.scatter(Z[:,0], Z[:,1], Z[:,2])
plt.title(f'DMAPS basi vectors (1,2,3)')
plt.savefig(path+"/3d_proj.png")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('2nd comp')
ax.set_ylabel('3rd comp')
ax.set_zlabel('4th comp')

ax.scatter(Z[:,1], Z[:,2], Z[:,3])
plt.title(f'DMAPS basi vectors (2,3,4)')
plt.savefig(path+"/3d_proj_2.png")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('1st comp')
ax.set_ylabel('2nd comp')
ax.set_zlabel('3rd comp')

ax.scatter(Z[:100,0], Z[:100,1], Z[:100,2])
plt.title(f'DMAPS basi vectors (1,2,3)')
plt.savefig(path+"/3d_proj_100.png")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('2nd comp')
ax.set_ylabel('3rd comp')
ax.set_zlabel('4th comp')

ax.scatter(Z[:100,1], Z[:100,2], Z[:100,3])
plt.title(f'DMAPS basi vectors (2,3,4)')
plt.savefig(path+"/3d_proj_2_100.png")