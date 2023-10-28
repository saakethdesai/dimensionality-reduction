import os
import numpy as np
import h5py
from matplotlib import pyplot as plt

time = np.array([0, 10, 20, 30, 40, 50])
r1_gt = np.array([51, 255, 411, 470, 476, 478])/512
r2_gt = np.array([31, 97, 117, 127, 133, 137])/512
r1_ae = np.array([47, 248, 420, 487, 496, 500])/512
r2_ae = np.array([30, 100, 122, 127, 130, 132])/512
r1_pca = np.array([49, 247, 403, 467, 474, 472])/512
r2_pca = np.array([29, 93, 115, 121, 125, 123])/512
r1_dmaps = np.array([49, 213, 394, 474, 472, 474])/512
r2_dmaps = np.array([29, 95, 103, 117, 137, 139])/512

cwd = os.getcwd()

fig = plt.figure(figsize=(8, 6))
sub5 = plt.subplot(1, 1, 1)
for axis in ['top', 'bottom', 'left', 'right']:
    sub5.spines[axis].set_linewidth(3)

# plotting the features
# r, rdf_gt_x, rdf_gt_d = calc_rdf(image_gt[index].reshape(1,image_size,image_size))

sub5.plot(time, r1_gt, lw=6, c='black', marker='o', markersize=12, linestyle='solid')
sub5.plot(time, r2_gt, lw=6, c='black', marker='o', markersize=12, linestyle='dashed')
#r, rdf_ae_x, rdf_ae_d = calc_rdf(image_ae[index].reshape(1,image_size,image_size))
sub5.plot(time, r1_ae, lw=2, c='#3773b8', marker='o', markersize=12, linestyle='solid')
sub5.plot(time, r2_ae, lw=2, c='#3773b8', marker='o', markersize=12, linestyle='dashed')
#r, rdf_pca_x, rdf_pca_d = calc_rdf(image_pca[index].reshape(1,image_size,image_size))
sub5.plot(time, r1_pca, lw=2, c='#e41a1c', marker='o', markersize=12, linestyle='solid')
sub5.plot(time, r2_pca, lw=2, c='#e41a1c', marker='o', markersize=12, linestyle='dashed')
#r, rdf_dmaps_x, rdf_dmaps_d = calc_rdf(image_dmaps[index].reshape(1,image_size,image_size))
sub5.plot(time, r1_dmaps, lw=2, c='#ff7f00', marker='o', markersize=12, linestyle='solid')
sub5.plot(time, r2_dmaps, lw=2, c='#ff7f00', marker='o', markersize=12, linestyle='dashed')

# Adding legends
colors = ['black', '#3773b8', '#e41a1c', '#ff7f00']
labels = ['GT', 'Autoencoder', 'PCA', 'Diffusion Maps']
lw = [6, 6, 4, 2]
for cc, col in enumerate(labels):
    sub5.plot(np.NaN, np.NaN, c=colors[cc], label=col, lw=lw[cc])

styles = ['-', '--']
ax2 = sub5.twinx()
for ss, sty in enumerate(styles):
    ax2.plot(np.NaN, np.NaN, ls=styles[ss],
             label=r'$r_{}$'.format(str(ss)), c='black')

ax2.get_yaxis().set_visible(False)

sub5.set_ylabel("relative absolute error", fontsize=25, labelpad=10)
sub5.set_xlabel("time", fontsize=25, labelpad=5)
#sub5.legend(fontsize=20)
sub5.tick_params(axis='both', which='major', labelsize=20)
sub5.tick_params(axis='both', which='minor', labelsize=10)
# sub5.set_ylim(-.01, .2)
# sub5.set_yscale('log')
sub5.legend(loc=0, fontsize=20)
ax2.legend(loc=2, fontsize=20)

fig.tight_layout()
plt.savefig("r_vs_t_dendrite.png")


# plotting the error
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

# Adding legends
colors = ['#3773b8', '#e41a1c', '#ff7f00']
labels = ['Autoencoder', 'PCA', 'Diffusion Maps']
lw = [6, 4, 2]
for cc, col in enumerate(labels):
    sub5.plot(np.NaN, np.NaN, c=colors[cc], label=col, lw=lw[cc])

styles = ['-', '--']
ax2 = sub5.twinx()
for ss, sty in enumerate(styles):
    ax2.plot(np.NaN, np.NaN, ls=styles[ss],
             label=r'$r_{}$'.format(str(ss)), c='black')

ax2.get_yaxis().set_visible(False)

sub5.set_ylabel("relative absolute error", fontsize=25, labelpad=10)
sub5.set_xlabel("time", fontsize=25, labelpad=5)
#sub5.legend(fontsize=20)
sub5.tick_params(axis='both', which='major', labelsize=20)
sub5.tick_params(axis='both', which='minor', labelsize=10)
sub5.set_ylim(-.01, .2)
# sub5.set_yscale('log')
sub5.legend(loc=0, fontsize=20)
ax2.legend(loc=2, fontsize=20)

fig.tight_layout()
plt.savefig("r_vs_t_dendrite_error.png")
