import os
import numpy as np
import h5py
from matplotlib import pyplot as plt

time = np.array([1, 4, 8, 12, 16, 20])
r_gt = np.array([0.062, 0.062, 0.061, 0.059, 0.062, 0.060])
r_ae = np.array([0.061, 0.062, 0.061, 0.059, 0.062, 0.061])
r_pca = np.array([0.063, 0.063, 0.061, 0.060, 0.063, 0.060])
r_dmaps = np.array([0.062, 0.062, 0.061, 0.060, 0.062, 0.060])

cwd = os.getcwd()

fig = plt.figure(figsize=(8, 6))
sub5 = plt.subplot(1, 1, 1)
for axis in ['top', 'bottom', 'left', 'right']:
    sub5.spines[axis].set_linewidth(3)

sub5.plot(time, r_gt, lw=6, c='black', marker='o', markersize=12, linestyle='solid')
sub5.plot(time, r_ae, lw=2, c='#3773b8', marker='o', markersize=12, linestyle='solid')
sub5.plot(time, r_pca, lw=2, c='#e41a1c', marker='o', markersize=12, linestyle='solid')
sub5.plot(time, r_dmaps, lw=2, c='#ff7f00', marker='o', markersize=12, linestyle='solid')

# Adding legends
colors = ['black', '#3773b8', '#e41a1c', '#ff7f00']
labels = ['GT', 'Autoencoder', 'PCA', 'Diffusion Maps']
lw = [6, 6, 4, 2]
styles = ['-', '-.', '-.', '-.']
for cc, col in enumerate(labels):
    sub5.plot(np.NaN, np.NaN, c=colors[cc], label=col, lw=lw[cc], ls=styles[cc])

# ax2 = sub5.twinx()
# for ss, sty in enumerate(styles):
#     ax2.plot(np.NaN, np.NaN, ls=styles[ss],
#              label='r' + str(ss), c='black')

sub5.legend(loc=0, fontsize=20)
# ax2.legend(loc=4, fontsize=20)
# ax2.get_yaxis().set_visible(False)


sub5.set_ylabel("feature measurements", fontsize=25, labelpad=10)
sub5.set_xlabel("time", fontsize=25, labelpad=5)
sub5.set_ylim(np.min([np.min(r_gt), 
                       np.min(r_ae), 
                       np.min(r_pca), 
                       np.min(r_dmaps)])-0.01,
            np.max([np.max(r_gt), 
                       np.max(r_ae), 
                       np.max(r_pca), 
                       np.max(r_dmaps)])+0.01
                )
# sub5.set_ylim(-.001, .03)
#sub5.legend(fontsize=20)
sub5.tick_params(axis='both', which='major', labelsize=20)
sub5.tick_params(axis='both', which='minor', labelsize=10)

fig.tight_layout()
plt.savefig("r_vs_t_gg.png")

cwd = os.getcwd()

fig = plt.figure(figsize=(8, 6))
sub5 = plt.subplot(1, 1, 1)
for axis in ['top', 'bottom', 'left', 'right']:
    sub5.spines[axis].set_linewidth(3)

sub5.plot(time, abs(r_ae-r_gt)/r_gt, lw=6, c='#3773b8', marker='o', markersize=12, linestyle='-.')
sub5.plot(time, abs(r_pca-r_gt)/r_gt, lw=4, c='#e41a1c', marker='o', markersize=12, linestyle='-.')
sub5.plot(time, abs(r_dmaps-r_gt)/r_gt, lw=2, c='#ff7f00', marker='o', markersize=12, linestyle='-.')

# Adding legends
colors = ['#3773b8', '#e41a1c', '#ff7f00']
labels = ['Autoencoder', 'PCA', 'Diffusion Maps']
lw = [6, 4, 2]
styles = ['-', '-.', '-.', '-.']
for cc, col in enumerate(labels):
    sub5.plot(np.NaN, np.NaN, c=colors[cc], label=col, lw=lw[cc], ls=styles[cc])

# ax2 = sub5.twinx()
# for ss, sty in enumerate(styles):
#     ax2.plot(np.NaN, np.NaN, ls=styles[ss],
#              label='r' + str(ss), c='black')

sub5.legend(loc=0, fontsize=20)
# ax2.legend(loc=4, fontsize=20)
# ax2.get_yaxis().set_visible(False)


sub5.set_ylabel("relative absolute error", fontsize=25, labelpad=10)
sub5.set_xlabel("time", fontsize=25, labelpad=5)
# sub5.set_ylim(np.min([np.min(r_gt), 
#                        np.min(r_ae), 
#                        np.min(r_pca), 
#                        np.min(r_dmaps)])-0.01,
#             np.max([np.max(r_gt), 
#                        np.max(r_ae), 
#                        np.max(r_pca), 
#                        np.max(r_dmaps)])+0.01
#                 )
sub5.set_ylim(-.001, .03)
#sub5.legend(fontsize=20)
sub5.tick_params(axis='both', which='major', labelsize=20)
sub5.tick_params(axis='both', which='minor', labelsize=10)

fig.tight_layout()
plt.savefig("r_vs_t_gg_error.png")
