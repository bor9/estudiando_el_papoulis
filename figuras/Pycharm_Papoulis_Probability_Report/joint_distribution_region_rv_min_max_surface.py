import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

from matplotlib import cm
from matplotlib import rc


__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

theta = 2
# number of samples between 0 and theta
np_theta = 200  # if 0 have index i, theta have index i+np_theta
dz = theta / np_theta  # sample period

z_min = -theta / 2
z_max = 2 * theta

# positive z
z_max_idx = math.floor(z_max/dz)
z = np.arange(z_max_idx+1, dtype=np.int16) * dz
# negative z
z_min_idx = math.floor(-z_min/dz)
z_neg = -np.arange(z_min_idx, 0, -1, dtype=np.int16) * dz

z = np.concatenate([z_neg, z])  # z values

# indexes of z=0 and z=theta
idx_0 = z_min_idx
idx_theta = z_min_idx + np_theta
z_len = z_max_idx + z_min_idx + 1

zz, ww = np.meshgrid(z, z)

# surface matrix
Fzw = np.zeros((z_len, z_len))

# region 0 < w < z < theta
idx_st = idx_0
for i in np.arange(idx_0, idx_theta):
    for j in np.arange(idx_st, idx_theta):
        Fzw[i, j] = (2 * zz[i, j] * ww[i, j] - (ww[i, j]**2)) / (theta**2)
    idx_st += 1
# region 0 < z < w < theta
idx_end = idx_0
for i in np.arange(idx_0, idx_theta):
    for j in np.arange(idx_0, idx_end):
        Fzw[i, j] = (zz[i, j]**2) / (theta**2)
    idx_end += 1
# region 0 < w < theta < z
for i in np.arange(idx_0, idx_theta):
    for j in np.arange(idx_theta, z_len):
        Fzw[i, j] = (2 * theta * ww[i, j] - (ww[i, j] ** 2)) / (theta ** 2)
# region 0 < z < theta < w
for i in np.arange(idx_theta, z_len):
    for j in np.arange(idx_0, idx_theta):
        Fzw[i, j] = (zz[i, j] ** 2) / (theta ** 2)
# region z > theta, w > theta
for i in np.arange(idx_theta, z_len):
    for j in np.arange(idx_theta, z_len):
        Fzw[i, j] = 1

# region limits
z_0_to_theta = z[idx_0: idx_theta]
r_lim1 = np.square(z_0_to_theta) / (theta**2)
z_theta = theta * np.ones(z_0_to_theta.shape)
r_lim2 = (2 * theta * z_0_to_theta - np.square(z_0_to_theta)) / (theta ** 2)
z_theta_to_end = z[idx_theta:-1]
theta * np.ones(z_theta_to_end.shape)

# z and w plot limits
z_ax_max = z_max + theta/2
z_ax_min = z_min - theta/2

# SURFACE PLOT
fig = plt.figure(0, figsize=(9, 4), frameon=False)
ax = fig.gca(projection='3d')
# plot the surface.
surf = ax.plot_surface(zz, ww, Fzw, cmap=cm.coolwarm, linewidth=0, antialiased=False, rstride=10, cstride=10)
# customize the z axis.
ax.set_xlim(z_ax_min, z_ax_max)
ax.set_ylim(z_ax_min, z_ax_max)
ax.set_zlim(0, 1)
ax.view_init(elev=58, azim=-118)

# axis arrows
arw = Arrow3D([z_ax_min, z_ax_max], [0, 0], [0, 0], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1, mutation_scale=5, facecolor='black')
ax.add_artist(arw)
arw = Arrow3D([0, 0], [z_ax_min, z_ax_max], [0, 0], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1, mutation_scale=5, facecolor='black')
ax.add_artist(arw)

ax.plot(z_0_to_theta, z_0_to_theta, r_lim1, 'k')
ax.plot(z_0_to_theta, theta * np.ones(z_0_to_theta.shape), r_lim1, 'k')
ax.plot(theta * np.ones(z_0_to_theta.shape), z_0_to_theta, r_lim2, 'k')
ax.plot(z_theta_to_end, theta * np.ones(z_theta_to_end.shape), 1, 'k')
ax.plot(theta * np.ones(z_theta_to_end.shape), z_theta_to_end, 1, 'k')

bl1 = -0.22
off1 = 0.2
bl2 = -0.25
off2 = 0.6
fontsize = 15
ax.text(z_ax_max+off1, 0, bl1, r'$z$', fontsize=fontsize, ha='center', va='baseline')
ax.text(theta+off1, 0, bl1, r'$\theta$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(0, z_ax_max+off2, bl2, r'$w$', fontsize=fontsize, ha='center', va='baseline')
ax.text(0, theta+off2, bl2, r'$\theta$', fontsize=fontsize, ha='center', va='baseline', color='k')

plt.axis('off')

# Add a color bar which maps values to colors.
cbaxes = inset_axes(ax, width="4%", height="60%", loc=6)
plt.colorbar(surf, cax=cbaxes, orientation='vertical')
# Distance view. Default is 10.
ax.dist = 8

plt.savefig('joint_distribution_region_rv_min_max_surface.pdf', bbox_inches='tight')
plt.show()

# CONTOUR PLOT

# z and w plot limits
z_ax_max = z_max+theta/10
z_ax_min = z_min+theta/10

fig = plt.figure(1, figsize=(5, 5), frameon=False)
plt.axis([z_ax_min, z_ax_max, z_ax_min, z_ax_max])
ax.set_aspect('equal', adjustable='box')
# axis arrows
plt.annotate("", xytext=(z_ax_min, 0), xycoords='data', xy=(z_ax_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0))
plt.annotate("", xytext=(0, z_ax_min), xycoords='data', xy=(0, z_ax_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0))

# plot contour with labels
levels = np.arange(0.1, 1, 0.1)
CS = plt.contour(zz, ww, Fzw, levels=levels, cmap=cm.coolwarm)
plt.clabel(CS, inline=1, fontsize=10, fmt="%1.1f")

# region limits
plt.plot([0, theta], [0, theta], 'k')
plt.plot([theta, theta], [0, z_max], 'k')
plt.plot([theta, z_max], [theta, theta], 'k')

plt.axis('off')
plt.savefig('joint_distribution_region_rv_min_max_contour.pdf', bbox_inches='tight')
plt.show()

# IMSHOW + CONTOUR PLOT

fig = plt.figure(2, figsize=(5, 5), frameon=False)
plt.axis([z_ax_min, z_ax_max, z_ax_min, z_ax_max])
ax.set_aspect('equal', adjustable='box')
# axis arrows
plt.annotate("", xytext=(z_ax_min, 0), xycoords='data', xy=(z_ax_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0))
plt.annotate("", xytext=(0, z_ax_min), xycoords='data', xy=(0, z_ax_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0))

im = plt.imshow(Fzw, interpolation='bilinear', origin='lower', cmap=cm.coolwarm, extent=(z_min, z_max, z_min, z_max))

# plot contour with labels
levels = np.arange(0.1, 1, 0.1)
CS = plt.contour(zz, ww, Fzw, levels=levels, colors='k')
plt.clabel(CS, inline=1, fontsize=10, fmt="%1.1f")

# region limits
plt.plot([0, theta], [0, theta], 'k')
plt.plot([theta, theta], [0, z_max], 'k')
plt.plot([theta, z_max], [theta, theta], 'k')

plt.axis('off')
plt.savefig('joint_distribution_region_rv_min_max_imshow.pdf', bbox_inches='tight')
plt.show()

# SURFACE + CONTOUR

# contour offset
offset = -1.5
z_ax_max = z_max + theta/2
z_ax_min = z_min - theta/2

fig = plt.figure(3, figsize=(7, 5), frameon=False)
ax = fig.gca(projection='3d')
# customize the z axis.
ax.set_xlim(z_ax_min, z_ax_max)
ax.set_ylim(z_ax_min, z_ax_max)
ax.set_zlim(offset, 1)
# plot the surface.
surf = ax.plot_surface(zz, ww, Fzw, cmap=cm.coolwarm, linewidth=0, antialiased=False, rstride=10, cstride=10)
ax.view_init(elev=34, azim=-118)

# axis arrows
arw = Arrow3D([z_ax_min, z_ax_max], [0, 0], [0, 0], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=10)
ax.add_artist(arw)
arw = Arrow3D([0, 0], [z_ax_min, z_ax_max], [0, 0], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=10)
ax.add_artist(arw)

ax.plot(z_0_to_theta, z_0_to_theta, r_lim1, 'k', zorder=10)
ax.plot(z_0_to_theta, theta * np.ones(z_0_to_theta.shape), r_lim1, 'k', zorder=10)
ax.plot(theta * np.ones(z_0_to_theta.shape), z_0_to_theta, r_lim2, 'k', zorder=10)
ax.plot(z_theta_to_end, theta * np.ones(z_theta_to_end.shape), 1, 'k', zorder=10)
ax.plot(theta * np.ones(z_theta_to_end.shape), z_theta_to_end, 1, 'k', zorder=10)
# labels
bl1 = -0.22
off1 = 0.2
bl2 = -0.25
off2 = 0.6
ax.text(z_ax_max+off1, 0, bl1, r'$z$', fontsize=fontsize, ha='center', va='baseline')
ax.text(theta+off1, 0, bl1, r'$\theta$', fontsize=fontsize, ha='center', va='baseline', color='k', zorder=10)
ax.text(0, z_ax_max+off2, bl2, r'$w$', fontsize=fontsize, ha='center', va='baseline')
ax.text(0, theta+off2, bl2, r'$\theta$', fontsize=fontsize, ha='center', va='baseline', color='k', zorder=10)

# Distance view. Default is 10.
ax.dist = 8

# Contour

# plot contour with labels
levels = np.arange(0.1, 1, 0.1)
CS = plt.contour(zz, ww, Fzw, levels=levels, cmap=cm.coolwarm, offset=offset, vmin=0, vmax=1)

# contour axis arrows
arw = Arrow3D([z_ax_min, z_ax_max], [0, 0], [offset, offset], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=10)
ax.add_artist(arw)
arw = Arrow3D([0, 0], [z_ax_min, z_ax_max], [offset, offset], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=10)
ax.add_artist(arw)

# region limits
plt.plot([0, theta], [0, theta], [offset, offset], 'k')
plt.plot([theta, theta], [0, z_max], [offset, offset], 'k')
plt.plot([theta, z_max], [theta, theta], [offset, offset], 'k')
plt.plot([0, theta], [theta, theta], [offset, offset], 'k')
# labels
bl1 = -0.22
off1 = 0.2
bl2 = -0.25
off2 = 0.6
ax.text(z_ax_max+off1, 0, offset+bl1, r'$z$', fontsize=fontsize, ha='center', va='baseline')
ax.text(theta+off1, 0, offset+bl1, r'$\theta$', fontsize=fontsize, ha='center', va='baseline', color='k', zorder=10)
ax.text(0, z_ax_max+off2, offset+bl2, r'$w$', fontsize=fontsize, ha='center', va='baseline')
ax.text(0, theta+off2, offset+bl2, r'$\theta$', fontsize=fontsize, ha='center', va='baseline', color='k', zorder=10)

plt.axis('off')

# Add a color bar which maps values to colors.
cbaxes = inset_axes(ax, width="4%", height="60%", loc=6)
cb = plt.colorbar(surf, cax=cbaxes, orientation='vertical')
cb.ax.tick_params(labelsize=9)

plt.savefig('joint_distribution_region_rv_min_max_surface_colormap.pdf', bbox_inches='tight')
plt.show()

print("z_max_original: {}".format(z_max))
print("z_max: {}".format(z[-1]))

print("z_min_original: {}".format(z_min))
print("z_min: {}".format(z_neg[0]))

print("z[idx_0]: {}".format(z[idx_0]))
print("z[idx_theta]: {}".format(z[idx_theta]))
