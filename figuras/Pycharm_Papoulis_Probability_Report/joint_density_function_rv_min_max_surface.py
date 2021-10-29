import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Polygon, Rectangle

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
fzw = np.zeros((z_len, z_len))

# region 0 < w < z < theta
idx_st = idx_0
for i in np.arange(idx_0, idx_theta):
    for j in np.arange(idx_st, idx_theta):
        fzw[i, j] = 2 / (theta ** 2)
    idx_st += 1

# region limits
z_0_to_theta = z[idx_0: idx_theta]
r_lim1 = np.square(z_0_to_theta) / (theta ** 2)
z_theta = theta * np.ones(z_0_to_theta.shape)
r_lim2 = (2 * theta * z_0_to_theta - np.square(z_0_to_theta)) / (theta ** 2)
z_theta_to_end = z[idx_theta:-1]
theta * np.ones(z_theta_to_end.shape)

# SURFACE + CONTOUR PLOT
# contour offset
offset = -1.5
# z and w plot limits
z_ax_max = z_max + theta/2
z_ax_min = z_min - theta/2
fontsize = 15
fig = plt.figure(0, figsize=(5, 5), frameon=False)
ax = fig.gca(projection='3d')
#ax = plt.subplot2grid((1, 10), (0, 3), rowspan=1, colspan=7, projection='3d')
# customize the z axis.
ax.set_xlim(z_ax_min, z_ax_max)
ax.set_ylim(z_ax_min, z_ax_max)
ax.set_zlim(offset, 1)
# plot the surface.
surf = ax.plot_surface(zz, ww, fzw, cmap=cm.coolwarm, linewidth=0, antialiased=False, rstride=10, cstride=10)
ax.view_init(elev=32, azim=-121)

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
CS = plt.contour(zz, ww, fzw, levels=levels, cmap=cm.coolwarm, offset=offset, vmin=0, vmax=1)

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
ax.text(z_ax_max+off1, 0, offset+bl1, r'$z$', fontsize=fontsize, ha='center', va='baseline')
ax.text(theta+off1, 0, offset+bl1, r'$\theta$', fontsize=fontsize, ha='center', va='baseline', color='k', zorder=10)
ax.text(0, z_ax_max+off2, offset+bl2, r'$w$', fontsize=fontsize, ha='center', va='baseline')
ax.text(0, theta+off2, offset+bl2, r'$\theta$', fontsize=fontsize, ha='center', va='baseline', color='k', zorder=10)

plt.axis('off')

# Add a color bar which maps values to colors.
cbaxes = inset_axes(ax, width="4%", height="60%", loc=6)
cb = plt.colorbar(surf, cax=cbaxes, orientation='vertical')
cb.ax.tick_params(labelsize=9)

plt.savefig('joint_density_function_rv_min_max_surface.pdf', bbox_inches='tight')
plt.show()


# PLANE REGIONS PLOT
z_min = -theta / 4
z_max = 2 * theta
fig = plt.figure(1, figsize=(5, 5), frameon=False)
ax = fig.add_subplot(111)
#ax = plt.subplot2grid((1, 10), (0, 0), rowspan=1, colspan=3)
# z and w plot limits
z_ax_max = z_max+theta/6
z_ax_min = z_min-theta/6

plt.axis([z_ax_min, z_ax_max, -5, z_ax_max])
ax.set_aspect('equal', adjustable='box')
# axis arrows
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(z_ax_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=5, headlength=6, facecolor='black', shrink=0))
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(0, z_ax_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=5, headlength=6, facecolor='black', shrink=0))

# region limits
plt.plot([0, theta], [0, theta], 'k', lw=2)
plt.plot([theta, theta], [0, z_max], 'k', lw=2)
plt.plot([theta, z_max], [theta, theta], 'k', lw=2)
plt.plot([0, theta], [theta, theta], 'k', lw=2)

c = np.linspace(0.8, 0.2, 6)
print(c)
# filled regions (z<0 or w<0)
vertices = np.array([[0, 0], [z_max, 0], [z_max, z_min], [z_min, z_min], [z_min, z_max], [0, z_max]])
ax.add_patch(Polygon(vertices, facecolor=c[0]*np.ones((3,)), edgecolor='none'))
vertices = np.array([[0, 0], [theta, 0], [theta, theta]])
ax.add_patch(Polygon(vertices, facecolor=c[1]*np.ones((3,)), edgecolor='none'))
vertices = np.array([[0, 0], [0, theta], [theta, theta]])
ax.add_patch(Polygon(vertices, facecolor=c[2]*np.ones((3,)), edgecolor='none'))
vertices = np.array([[0, theta], [theta, theta], [theta, z_max], [0, z_max]])
ax.add_patch(Polygon(vertices, facecolor=c[3]*np.ones((3,)), edgecolor='none'))
vertices = np.array([[theta, 0], [z_max, 0], [z_max, theta], [theta, theta]])
ax.add_patch(Polygon(vertices, facecolor=c[4]*np.ones((3,)), edgecolor='none'))
vertices = np.array([[theta, theta], [z_max, theta], [z_max, z_max], [theta, z_max]])
ax.add_patch(Polygon(vertices, facecolor=c[5]*np.ones((3,)), edgecolor='none'))

# axis labels
bl3 = -0.45
plt.text(z_ax_max, bl3, r'$z$', fontsize=fontsize, ha='center', va='baseline')
plt.text(theta, bl3, r'$\theta$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-0.1, z_ax_max, r'$w$', fontsize=fontsize, ha='right', va='center')
plt.text(-0.1, theta, r'$\theta$', fontsize=fontsize, ha='right', va='center')


# legend
handles = []
labels = [r'$(z<0)\cup(w<0):\,0$',
          r'$0\leq w<z<\theta:\,(2zw-w^2)/\theta^2$',
          r'$0\leq z\leq w<\theta:\,z^2/\theta^2$',
          r'$0\leq z<\theta\leq w:\,z^2/\theta^2$',
          r'$0\leq w<\theta\leq z:(2\theta w-w^2)/\theta^2$',
          r'$(z>\theta)\cap(w>\theta):\,1$']
for ci in c:
    handles.append(Rectangle((0, 0), 1, 1, color=ci*np.ones((3,)), linewidth=0))
leg = plt.legend(handles=handles,
                 labels=labels,
                 framealpha=1,
                 loc='center left',
                 bbox_to_anchor=(-0.17, 0.23),
                 fontsize=12,
                 frameon=False)

plt.axis('off')

#plt.savefig('joint_distribution_region_rv_min_max_surface_colormap_v3.pdf', bbox_inches='tight')
#plt.show()

print("z_max_original: {}".format(z_max))
print("z_max: {}".format(z[-1]))

print("z_min_original: {}".format(z_min))
print("z_min: {}".format(z_neg[0]))

print("z[idx_0]: {}".format(z[idx_0]))
print("z[idx_theta]: {}".format(z[idx_theta]))
