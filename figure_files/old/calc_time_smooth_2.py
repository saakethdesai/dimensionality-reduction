import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as sp
from scipy.misc import derivative
from scipy import interpolate
from matplotlib import rc

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

        ax.bar(np.arange(len(nder_change_sign)), nder_change_sign, color="blue", label="Number of changes in \n gradient sign, " + r'$N(t)$', width=2.0)
        #ax.set_title(name, fontsize=30)

    return D

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

#rc('text', usetex=True)

# fig, axs = plt.subplots(2,3, figsize=(15, 10))
#fig = plt.figure(figsize=(30, 5))
#fig.subplots_adjust(wspace=.4, hspace=.3)
#ax1 = fig.add_subplot(1, 3, 1, projection='3d')
#ax2 = fig.add_subplot(1, 3, 2)
#ax3 = fig.add_subplot(1, 3, 3)

#axs = [ax1, ax2, ax3]
#axs_twin = [None]+ [ax.twinx() for ax in axs[1:]]

ax = plt.figure().add_subplot(projection='3d')

# 3D plot 
#raw plot
ax.plot(x_traj, y_traj, z_traj1 , color='k', linewidth=5, label="smooth")
ax.plot(x_traj, y_traj, z_traj2 , color='orange', linewidth=5, label="Non smooth")
#set labels
ax.set_xlabel(r"$x$", fontsize=25, labelpad=15)
ax.set_ylabel(r"$y$", fontsize=25, labelpad=15)
ax.set_zlabel(r"$z$", fontsize=25, labelpad=10)
#fiddle with margins to get little overlap
ax.axes.set_xlim3d(left=-0.99, right=1.01) 
ax.axes.set_ylim3d(bottom=-0.99, top=1.00) 
ax.axes.set_zlim3d(bottom=0.0, top=6.2) 
#set tick labels and sizes
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.zaxis.set_tick_params(labelsize=20)
ax.set_xticks([-1,0,1])
ax.set_yticks([-1,0,1])
ax.set_zticks([0,3,6])
#axs[0].grid(False)
#set thick axis lines
for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
    axis.line.set_linewidth(3)
#make thin grid lines
ax.w_xaxis.gridlines.set_lw(0.5)
ax.w_yaxis.gridlines.set_lw(0.5)
ax.w_zaxis.gridlines.set_lw(0.5)
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#axs[0].legend(fontsize = 25,loc="upper right", bbox_to_anchor=(1.7, 1.1))

index_list = [25, 50, 75]
for index in index_list:
    ax.scatter(x_traj[index], y_traj[index], z_traj1[index], marker="*", s=400, c='k')
    ax.scatter(x_traj[index], y_traj[index], z_traj2[index], marker="*", s=400, c='orange')


#fig.tight_layout()
plt.savefig("test0.png",  bbox_inches = 'tight',
    pad_inches = 0)

grad_x = np.gradient(x_traj)
grad_y = np.gradient(y_traj)
grad_z = np.gradient(z_traj2)


#index = 10
#for i in [45,55]:
#    axs[0].quiver(x_traj[i]
#                , y_traj[i]
#                , z_traj2[i]
#                , grad_x[i]
#                , grad_y[i]
#                , grad_z[i]
#                , pivot = 'middle'
#                , length=20,linewidths=2, arrow_length_ratio=0.2, label="direction")

#    axs[0].scatter(x_traj[i],
#               y_traj[i],
#               z_traj2[i])

#1st plot

ax = plt.figure().add_subplot()
ax_twin = ax.twinx() 

smooth = np.concatenate((
                        x_traj.reshape(-1,1),
                        y_traj.reshape(-1,1), 
                        z_traj1.reshape(-1,1)
                        ), 
                        axis=1)

constant_direction(ax, smooth, [0,-1], "smooth")
slope_direct(ax_twin, smooth, "smooth")

#constant_direction(ax, Nsmooth, [0,-1], "Non smooth")
#slope_direct(axs_twin[2], Nsmooth, "Non smooth")

ax.set_ylabel(r"$N(t)$", fontsize=25, color='blue')
ax.set_xlabel("Time", fontsize=25)
ax.set_xticks([0,25,50,75,100])
ax.set_ylim([0.0, 1])
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=10)
#ax.legend(fontsize = 25, loc="upper left")

ax.tick_params(axis='y', colors='blue')
ax_twin.tick_params(axis='y', colors='red')

ax_twin.set_ylabel(r'$\theta(t)$', fontsize=25, color='red')
ax_twin.set_ylim([0.0, np.pi/2])
ax_twin.tick_params(axis='both', which='major', labelsize=20)
ax_twin.tick_params(axis='both', which='minor', labelsize=10)
#ax_twin[i].legend(fontsize = 25, loc="upper left", bbox_to_anchor=(0, .75))
    
ax.scatter(24, 0.38, marker="*", s=400, c='black')
ax.scatter(49, 0.38, marker="*", s=400, c='black')
ax.scatter(74, 0.38, marker="*", s=400, c='black')

#fig.tight_layout()
plt.savefig("test1.png",  bbox_inches = 'tight',
    pad_inches = 0)

#2nd plot

ax = plt.figure().add_subplot()
ax_twin = ax.twinx() 

Nsmooth = np.concatenate((
                        x_traj.reshape(-1,1),
                        y_traj.reshape(-1,1), 
                        z_traj2.reshape(-1,1)
                        ),
                        axis=1)


constant_direction(ax, Nsmooth, [0,-1], "Non smooth")
slope_direct(ax_twin, Nsmooth, "Non smooth")

ax.set_ylabel(r"$N(t)$", fontsize=25, color='blue')
ax.set_xlabel("Time", fontsize=25)
ax.set_xticks([0,25,50,75,100])
ax.set_ylim([0.0, 1])
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=10)
#ax.legend(fontsize = 25, loc="upper left")

ax.tick_params(axis='y', colors='blue')
ax_twin.tick_params(axis='y', colors='red')

ax_twin.set_ylabel(r'$\theta(t)$', fontsize=25, color='red')
ax_twin.set_ylim([0.0, np.pi/2])
ax_twin.tick_params(axis='both', which='major', labelsize=20)
ax_twin.tick_params(axis='both', which='minor', labelsize=10)

ax.scatter(24, 0.38, marker="*", s=400, c='orange')
ax.scatter(49, 0.72, marker="*", s=400, c='orange')
ax.scatter(74, 0.38, marker="*", s=400, c='orange')

# -----------------------------------------------------------------------------
#fig.tight_layout()
plt.savefig("test2.png",  bbox_inches = 'tight',
    pad_inches = 0)
