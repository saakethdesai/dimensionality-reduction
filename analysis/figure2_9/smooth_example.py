import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as sp
from scipy.misc import derivative
from scipy import interpolate


def slope_direct(ax, z, name):

    # grad = np.gradient(z, axis=0)

    t = np.arange(len(z))
    grad = np.zeros([len(z),0])
    for j in range(z.shape[1]):
        cs = interpolate.CubicSpline(t,  z[:,j])
        dx = derivative(cs, t)
        # dx = derivative(lambda i: np.interp(i, t, z[:,j]), t)
        
        grad = np.append(grad, dx.reshape(-1,1),
                         axis=1)
    
    a = np.diag(np.dot(grad[1:], grad[:-1].T))
    print(grad.shape)

    a = 1-np.diag(sp.distance.cdist(grad[1:], grad[:-1], 'cosine'))
    
    ax.plot(np.arange(len(a)), np.arccos(a), color="red", label=r"Average change" + "\n" + r"in direction", linewidth=5)

    return np.arccos(a)

def constant_direction(ax, z, start, name=""):

    D = z.shape[1]

    for j in range(1):
        xx = z[start[j]:start[j+1], :] # 50
        # [N , D]
        xx_grad = np.gradient(xx, axis=0) # (50 ,D)
        xx_grad_sign = np.sign(xx_grad) #(50, D)
        nder_change_sign = []
        for k in range(xx_grad_sign.shape[0]-1):
            orig_sign = xx_grad_sign[k, :] #(1, D)
            new_sign = xx_grad_sign[k+1, :]
            count = np.count_nonzero(new_sign - orig_sign)
            nder_change_sign.append(count/D)

        ax.bar(np.arange(len(nder_change_sign)), nder_change_sign, color="blue", label="Number of changes in \n gradient sign, " + r'$N(t)$')
        ax.set_title(name, fontsize=30)

    return D

# fig, axs = plt.subplots(2,3, figsize=(15, 10))
fig = plt.figure(figsize=(30, 5))
fig.subplots_adjust(wspace=.4, hspace=.3)
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

axs = [ax1, ax2, ax3]
axs_twin = [None]+ [ax.twinx() for ax in axs[1:]]

# -----------------------------------------------------------------------------

# Define the parametric equations for the trajectory
def x(t):
    return np.sin(t) #+ .1*np.random.normal(0,1, len(t))

def y(t):
    return np.cos(t) #+ .1*np.random.normal(0,1, len(t))

def z1(t):
    return t #+ .1*np.random.normal(0,1, len(t)) #np.zeros(len(t)) #np.abs(t-np.pi) ** (2/3)

def z2(t):
    return np.abs(t-np.pi) #+ .1*np.random.normal(0,1, len(t)) #np.zeros(len(t)) #np.abs(t-np.pi) ** (2/3)

# Define the range of the parameter t
t = np.linspace(0, 2*np.pi, 100)

# Evaluate the trajectory at the values of t
x_traj = x(t)
y_traj = y(t)
z_traj1 = z1(t)
z_traj2 = z2(t)

# Create a 3D plot of the trajectory
axs[0].plot(x_traj, y_traj, z_traj1 , color='k', linewidth=2, label="smooth")
axs[0].plot(x_traj, y_traj, z_traj2 , color='g', linewidth=2, label="Non smooth")
axs[0].xaxis.set_ticklabels([])
axs[0].yaxis.set_ticklabels([])
axs[0].zaxis.set_ticklabels([])
# axs[0].set_xlabel('X')
# axs[0].set_ylabel('Y')
# axs[0].set_zlabel('Z')
axs[0].legend(fontsize = 25,loc="upper right", bbox_to_anchor=(1.7, 1.1))

grad_x = np.gradient(x_traj)
grad_y = np.gradient(y_traj)
grad_z = np.gradient(z_traj2)

index = 10
for i in [45,55]:
    axs[0].quiver(x_traj[i]
                , y_traj[i]
                , z_traj2[i]
                , grad_x[i]
                , grad_y[i]
                , grad_z[i]
                , pivot = 'middle'
                , length=20,linewidths=2, arrow_length_ratio=0.2, label="direction")

    axs[0].scatter(x_traj[i],
               y_traj[i],
               z_traj2[i])

smooth = np.concatenate((
                        x_traj.reshape(-1,1),
                        y_traj.reshape(-1,1), 
                        z_traj1.reshape(-1,1)
                        ), 
                        axis=1)

Nsmooth = np.concatenate((
                        x_traj.reshape(-1,1),
                        y_traj.reshape(-1,1), 
                        z_traj2.reshape(-1,1)
                        ),
                        axis=1)

constant_direction(axs[1], smooth, [0,-1], "smooth")
slope_direct(axs_twin[1], smooth, "smooth")
constant_direction(axs[2], Nsmooth, [0,-1], "Non smooth")
slope_direct(axs_twin[2], Nsmooth, "Non smooth")

for i in range(1,3):
    axs[i].set_ylabel(r"$N(t)$", fontsize=25)
    axs[i].set_xlabel("Time", fontsize=25)
    axs[i].set_ylim([0.0, 1])
    axs[i].tick_params(axis='both', which='major', labelsize=25)
    axs[i].tick_params(axis='both', which='minor', labelsize=10)
    axs[i].legend(fontsize = 25, loc="upper left")

    axs_twin[i].set_ylabel(r'$\theta(t)$', fontsize=25)
    axs_twin[i].set_ylim([0.0, np.pi/2])
    axs_twin[i].tick_params(axis='both', which='major', labelsize=25)
    axs_twin[i].tick_params(axis='both', which='minor', labelsize=10)
    axs_twin[i].legend(fontsize = 25, loc="upper left", bbox_to_anchor=(0, .75))

axs[0].dist = 7

# -----------------------------------------------------------------------------
fig.tight_layout()
plt.savefig("smooth_example.png",  bbox_inches = 'tight',
    pad_inches = 0)