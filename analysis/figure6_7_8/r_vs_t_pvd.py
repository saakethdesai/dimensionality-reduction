import os
import numpy as np
import h5py
from matplotlib import pyplot as plt

cwd = os.getcwd()

fig = plt.figure(figsize=(8, 6))
sub5 = plt.subplot(1, 1, 1)
for axis in ['top', 'bottom', 'left', 'right']:
    sub5.spines[axis].set_linewidth(3)

time = np.array([10, 20, 30, 40, 50])
#r, rdf_gt_x, rdf_gt_d = calc_rdf(image_gt[index].reshape(1,image_size,image_size))
r1_gt = np.array([8.5, 8.5, 10.3, 11.2, 12.3])/256
r2_gt = np.array([15.6, 15.6, 19.1, 26.6, 26.6])/256
#r, rdf_ae_x, rdf_ae_d = calc_rdf(image_ae[index].reshape(1,image_size,image_size))
r1_ae = np.array([8.7, 8.7, 10.3, 11.2, 12.3])/256
r2_ae = np.array([15.8, 15.8, 18.7, 26.6, 26.6])/256
#r, rdf_pca_x, rdf_pca_d = calc_rdf(image_pca[index].reshape(1,image_size,image_size))
r1_pca = np.array([9.4, 9.4, 11.2, 11.8, 12.3])/256
r2_pca = np.array([16.5, 16.5, 19.2, 26.6, 26.6])/256
#r, rdf_dmaps_x, rdf_dmaps_d = calc_rdf(image_dmaps[index].reshape(1,image_size,image_size))
r1_dmaps = np.array([8.8, 8.8, 11.0, 11.6, 12.3])/256
r2_dmaps = np.array([15.8, 15.8, 19.2, 26.6, 26.6])/256

fig = plt.figure(figsize=(8, 6))
sub5 = plt.subplot(1, 1, 1)
for axis in ['top', 'bottom', 'left', 'right']:
    sub5.spines[axis].set_linewidth(3)

sub5.plot(time, r1_gt, lw=6, c='black', marker='o', markersize=12, linestyle='solid')
sub5.plot(time, r2_gt, lw=6, c='black', marker='o', markersize=12, linestyle='dashed')

sub5.plot(time, r1_ae, lw=6, c='#3773b8', marker='o', markersize=12, linestyle='solid')
sub5.plot(time, r2_ae, lw=6, c='#3773b8', marker='o', markersize=12, linestyle='dashed')

sub5.plot(time, r1_pca, lw=4, c='#e41a1c', marker='o', markersize=12, linestyle='solid')
sub5.plot(time, r2_pca, lw=4, c='#e41a1c', marker='o', markersize=12, linestyle='dashed')

sub5.plot(time, r1_dmaps, lw=2, c='#ff7f00', marker='o', markersize=12, linestyle='solid')
sub5.plot(time, r2_dmaps, lw=2, c='#ff7f00', marker='o', markersize=12, linestyle='dashed')

# Adding legends
colors = ['black', '#3773b8', '#e41a1c', '#ff7f00']
labels = ['GT', 'Autoencoder', 'PCA', 'Diffusion Maps']
lw = [6, 6, 4, 2]
for cc, col in enumerate(labels):
    sub5.plot(np.NaN, np.NaN, c=colors[cc], label=col, lw=lw[cc])

ax2 = sub5.twinx()
styles = ['-', '--']
for ss, sty in enumerate(styles):
    ax2.plot(np.NaN, np.NaN, ls=styles[ss],
             label=r'$r_{}$'.format(str(ss)), c='black')

ax2.get_yaxis().set_visible(False)

sub5.legend(loc=0, fontsize=20)
# sub5.set_ylim(-.01, .2)
ax2.legend(loc=4, fontsize=20)
# sub5.set_yscale('log')
fig.tight_layout()
plt.savefig("r_vs_t_pvd.png")

# # ---------------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(8, 6))
sub5 = plt.subplot(1, 1, 1)
for axis in ['top', 'bottom', 'left', 'right']:
    sub5.spines[axis].set_linewidth(3)

sub5.plot(time, abs(r1_ae-r1_gt)/r1_gt, lw=6, c='#3773b8', marker='o', markersize=12, linestyle='solid')
sub5.plot(time, abs(r2_ae-r2_gt)/r2_gt, lw=6, c='#3773b8', marker='o', markersize=12, linestyle='dashed')


sub5.plot(time, abs(r1_pca-r1_gt)/r1_gt, lw=4, c='#e41a1c', marker='o', markersize=12, linestyle='solid')
sub5.plot(time, abs(r2_pca-r2_gt)/r2_gt, lw=4, c='#e41a1c', marker='o', markersize=12, linestyle='dashed')


sub5.plot(time, abs(r1_dmaps-r1_gt)/r1_gt, lw=2, c='#ff7f00', marker='o', markersize=12, linestyle='solid')
sub5.plot(time, abs(r2_dmaps-r2_gt)/r2_gt, lw=2, c='#ff7f00', marker='o', markersize=12, linestyle='dashed')

sub5.set_ylabel("relative absolute error", fontsize=25, labelpad=10)
sub5.set_xlabel("time", fontsize=25, labelpad=5)
#sub5.legend(fontsize=20)
sub5.tick_params(axis='both', which='major', labelsize=20)
sub5.tick_params(axis='both', which='minor', labelsize=10)

# Adding legends
colors = ['#3773b8', '#e41a1c', '#ff7f00']
labels = ['Autoencoder', 'PCA', 'Diffusion Maps']
lw = [6, 4, 2]
for cc, col in enumerate(labels):
    sub5.plot(np.NaN, np.NaN, c=colors[cc], label=col, lw=lw[cc])

ax2 = sub5.twinx()
styles = ['-', '--']
for ss, sty in enumerate(styles):
    ax2.plot(np.NaN, np.NaN, ls=styles[ss],
             label=r'$r_{}$'.format(str(ss)), c='black')

ax2.get_yaxis().set_visible(False)

sub5.legend(loc=0, fontsize=20)
sub5.set_ylim(-.01, .2)
ax2.legend(loc=2, fontsize=20)
# sub5.set_yscale('log')
fig.tight_layout()
plt.savefig("r_vs_t_pvd_error.png")