import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Polygon

import matplotlib.colors as colors

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
z_min = -theta / 2
z_max = 2 * theta
fzw = 2 / (theta ** 2)
fzw = 0.08

# SURFACE + CONTOUR PLOT
# contour offset
offset = -0.2
# z and w plot limits
z_ax_max = z_max + theta/2
z_ax_min = z_min - theta/2
fontsize = 15

fig = plt.figure(0, figsize=(5, 4), frameon=False)
ax = fig.gca(projection='3d')
# customize the z axis.
ax.set_xlim(z_ax_min, z_ax_max)
ax.set_ylim(z_ax_min, z_ax_max)
ax.set_zlim(offset, fzw)
ax.view_init(elev=33, azim=-149)

# plot the surface.
# axis arrows
arw = Arrow3D([z_ax_min, z_ax_max], [0, 0], [0, 0], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=10)
ax.add_artist(arw)
arw = Arrow3D([0, 0], [z_ax_min, z_ax_max], [0, 0], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=10)
ax.add_artist(arw)

# Density surface
F = [[z_min, z_min, 0],
     [z_max, z_min, 0],
     [z_max, z_max, 0],
     [z_min, z_max, 0]
     ]
T1 = [[0, 0, 0],
      [theta, 0, 0],
      [theta, 0, fzw],
      [0, 0, fzw],
      ]
T2 = [[theta, 0, 0],
      [theta, theta, 0],
      [theta, theta, fzw],
      [theta, 0, fzw],
      ]
T3 = [[theta, theta, 0],
      [theta, theta, fzw],
      [0, 0, fzw],
      [0, 0, 0],
      ]
C = [[0, 0, fzw],
     [theta, 0, fzw],
     [theta, theta, fzw],
     ]

# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col1 = scalarMap.to_rgba(0)
col2 = scalarMap.to_rgba(1)

coll = Poly3DCollection([F, T1, T2, T3, C], facecolors=[col1, col2, col2, col2, col2],
                        edgecolors=[col1, 'k', 'k', 'k', 'k'])
fig.gca().add_collection(coll)

# labels
off1 = -0.45
off2 = -0.7
ax.text(z_ax_max, off1, 0, r'$z$', fontsize=fontsize, ha='right', va='center')
ax.text(theta, off1, 0, r'$\theta$', fontsize=fontsize, ha='center', va='center', color='k',
        zorder=10)
ax.text(off2, z_ax_max, 0, r'$w$', fontsize=fontsize, ha='left', va='center')
ax.text(off2, theta, 0, r'$\theta$', fontsize=fontsize, ha='left', va='center', color='k',
        zorder=10)

# region limits
plt.plot([theta, theta], [theta, z_max], [0, 0], 'k', zorder=9)
# tick
plt.plot([0, 0.2], [theta, theta], [0, 0], 'k', lw=1, zorder=9)

# Distance view. Default is 10.
ax.dist = 8

# Projection
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
# tick
plt.plot([0, 0.2], [theta, theta], [offset, offset], 'k', lw=1)

#Floor
F = [[z_min, z_min, offset],
     [z_max, z_min, offset],
     [z_max, z_max, offset],
     [z_min, z_max, offset]
     ]
F2 = [[0, 0, offset],
      [theta, 0, offset],
      [theta, theta, offset],
      ]

coll = Poly3DCollection([F, F2], facecolors=[col1, col2], edgecolors=[col1, 'k'])
fig.gca().add_collection(coll)

# labels
ax.text(z_ax_max, off1, offset, r'$z$', fontsize=fontsize, ha='right', va='center')
ax.text(theta, off1, offset, r'$\theta$', fontsize=fontsize, ha='center', va='center', color='k',
        zorder=10)
ax.text(off2, z_ax_max, offset, r'$w$', fontsize=fontsize, ha='left', va='center')
ax.text(off2, theta, offset, r'$\theta$', fontsize=fontsize, ha='left', va='center', color='k',
        zorder=10)

plt.axis('off')
plt.savefig('joint_density_function_rv_min_max_surface_v2.pdf', bbox_inches='tight', pad_inches=0)
#plt.show()


# INTEGRATION REGIONS PLOT
z_min = -theta / 3
z_max = 2 * theta
# z and w plot limits
z_ax_max = z_max+theta/6
z_ax_min = z_min-theta/6

col3 = scalarMap.to_rgba(0.5)
z_bl = -0.37  # baseline of z axis labels
w_mg = -0.1 # right margin of w axis labels
t_center = (z_ax_min + z_ax_max)/2
t_bl = -1.2

fig = plt.figure(1, figsize=(8, 8), frameon=False)
#
# SUBPLOT 1
#
ax = plt.subplot2grid((4, 4), (0, 0), rowspan=2, colspan=2)
plt.axis([z_ax_min, z_ax_max, z_ax_min, z_ax_max])
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
# tick
plt.plot([0, 0.1], [theta, theta], 'k', lw=1)

# plane regions
vertices = np.array([[z_min, z_min], [z_max, z_min], [z_max, z_max], [z_min, z_max]])
ax.add_patch(Polygon(vertices, facecolor=col1, edgecolor='none'))
vertices = np.array([[0, 0], [theta, 0], [theta, theta]])
ax.add_patch(Polygon(vertices, facecolor=col2, edgecolor='none'))

z1 = 2* theta / 3
w1 = theta / 3

plt.plot(z1, w1, 'k.', markersize=6)
plt.plot([z1, z1], [0, w1], 'k', lw=1)
plt.plot([0, z1], [w1, w1], 'k', lw=1)

# integration region
vertices = np.array([[0, 0], [z1, 0], [z1, w1], [w1, w1]])
ax.add_patch(Polygon(vertices, facecolor=col3, edgecolor='none', alpha=0.5))

# axis labels
plt.text(z_ax_max, z_bl, r'$z$', fontsize=fontsize, ha='center', va='baseline')
plt.text(theta, z_bl, r'$\theta$', fontsize=fontsize, ha='center', va='baseline')
plt.text(z1, z_bl, r'$z_1$', fontsize=fontsize, ha='center', va='baseline')
plt.text(0, z_bl, r'$0$', fontsize=fontsize, ha='center', va='baseline')
plt.text(w_mg, z_ax_max, r'$w$', fontsize=fontsize, ha='right', va='center')
plt.text(w_mg, theta, r'$\theta$', fontsize=fontsize, ha='right', va='center')
plt.text(w_mg, w1, r'$w_1$', fontsize=fontsize, ha='right', va='center')

plt.text(t_center, t_bl, r'$0\leq w_1<z_1<\theta$', fontsize=fontsize, ha='center', va='baseline')
plt.axis('off')

#
# SUBPLOT 2
#
ax = plt.subplot2grid((4, 4), (0, 2), rowspan=2, colspan=2)
plt.axis([z_ax_min, z_ax_max, z_ax_min, z_ax_max])
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
# tick
plt.plot([0, 0.1], [theta, theta], 'k', lw=1)

# plane regions
vertices = np.array([[z_min, z_min], [z_max, z_min], [z_max, z_max], [z_min, z_max]])
ax.add_patch(Polygon(vertices, facecolor=col1, edgecolor='none'))
vertices = np.array([[0, 0], [theta, 0], [theta, theta]])
ax.add_patch(Polygon(vertices, facecolor=col2, edgecolor='none'))

z2 = theta / 3
w2 = 5 * theta / 3

plt.plot(z2, w2, 'k.', markersize=6)
plt.plot([z2, z2], [0, w2], 'k', lw=1)
plt.plot([0, z2], [w2, w2], 'k', lw=1)
plt.text(z2, w2, r'$(z_2,\,w_2)$', fontsize=fontsize, ha='left', va='bottom')


# integration region
vertices = np.array([[0, 0], [z2, 0], [z2, z2]])
ax.add_patch(Polygon(vertices, facecolor=col3, edgecolor='none', alpha=0.5))

# axis labels
plt.text(z_ax_max, z_bl, r'$z$', fontsize=fontsize, ha='center', va='baseline')
plt.text(theta, z_bl, r'$\theta$', fontsize=fontsize, ha='center', va='baseline')
plt.text(z2, z_bl, r'$z_2$', fontsize=fontsize, ha='center', va='baseline')
plt.text(0, z_bl, r'$0$', fontsize=fontsize, ha='center', va='baseline')
plt.text(w_mg, z_ax_max, r'$w$', fontsize=fontsize, ha='right', va='center')
plt.text(w_mg, theta, r'$\theta$', fontsize=fontsize, ha='right', va='center')
plt.text(w_mg, w2, r'$w_2$', fontsize=fontsize, ha='right', va='center')

plt.text(t_center, t_bl, r'$0\leq z_2<\theta\leq w_2$', fontsize=fontsize, ha='center', va='baseline')
plt.axis('off')
#
# SUBPLOT
#
ax = plt.subplot2grid((4, 4), (2, 0), rowspan=2, colspan=2)
plt.axis([z_ax_min, z_ax_max, z_ax_min, z_ax_max])
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
# tick
plt.plot([0, 0.1], [theta, theta], 'k', lw=1)

# plane regions
vertices = np.array([[z_min, z_min], [z_max, z_min], [z_max, z_max], [z_min, z_max]])
ax.add_patch(Polygon(vertices, facecolor=col1, edgecolor='none'))
vertices = np.array([[0, 0], [theta, 0], [theta, theta]])
ax.add_patch(Polygon(vertices, facecolor=col2, edgecolor='none'))

z3 = 4 * theta / 3
w3 = theta / 3

plt.plot(z3, w3, 'k.', markersize=6)
plt.plot([z3, z3], [0, w3], 'k', lw=1)
plt.plot([0, z3], [w3, w3], 'k', lw=1)
plt.text(z3, w3, r'$(z_3,\,w_3)$', fontsize=fontsize, ha='left', va='bottom')

# integration region
vertices = np.array([[0, 0], [theta, 0], [theta, w3], [w3, w3]])
ax.add_patch(Polygon(vertices, facecolor=col3, edgecolor='none', alpha=0.5))

# axis labels
plt.text(z_ax_max, z_bl, r'$z$', fontsize=fontsize, ha='center', va='baseline')
plt.text(theta, z_bl, r'$\theta$', fontsize=fontsize, ha='center', va='baseline')
plt.text(z3, z_bl, r'$z_3$', fontsize=fontsize, ha='center', va='baseline')
plt.text(0, z_bl, r'$0$', fontsize=fontsize, ha='center', va='baseline')
plt.text(w_mg, z_ax_max, r'$w$', fontsize=fontsize, ha='right', va='center')
plt.text(w_mg, theta, r'$\theta$', fontsize=fontsize, ha='right', va='center')
plt.text(w_mg, w3, r'$w_3$', fontsize=fontsize, ha='right', va='center')

plt.text(t_center, t_bl, r'$0\leq w_3<\theta\leq z_3$', fontsize=fontsize, ha='center', va='baseline')
plt.axis('off')
#
# SUBPLOT
#
ax = plt.subplot2grid((4, 4), (2, 2), rowspan=2, colspan=2)
plt.axis([z_ax_min, z_ax_max, z_ax_min, z_ax_max])
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
# tick
plt.plot([0, 0.1], [theta, theta], 'k', lw=1)

# plane regions
vertices = np.array([[z_min, z_min], [z_max, z_min], [z_max, z_max], [z_min, z_max]])
ax.add_patch(Polygon(vertices, facecolor=col1, edgecolor='none'))
vertices = np.array([[0, 0], [theta, 0], [theta, theta]])
ax.add_patch(Polygon(vertices, facecolor=col2, edgecolor='none'))

z4 = 5 * theta / 4
w4 = theta + 0.8

plt.plot(z4, w4, 'k.', markersize=6)
plt.plot([z4, z4], [0, w4], 'k', lw=1)
plt.plot([0, z4], [w4, w4], 'k', lw=1)
plt.text(z4, w4, r'$(z_4,\,w_4)$', fontsize=fontsize, ha='left', va='bottom')

# integration region
vertices = np.array([[0, 0], [theta, 0], [theta, theta]])
ax.add_patch(Polygon(vertices, facecolor=col3, edgecolor='none', alpha=0.5))

# axis labels
plt.text(z_ax_max, z_bl, r'$z$', fontsize=fontsize, ha='center', va='baseline')
plt.text(theta, z_bl, r'$\theta$', fontsize=fontsize, ha='center', va='baseline')
plt.text(z4, z_bl, r'$z_4$', fontsize=fontsize, ha='center', va='baseline')
plt.text(0, z_bl, r'$0$', fontsize=fontsize, ha='center', va='baseline')
plt.text(w_mg, z_ax_max, r'$w$', fontsize=fontsize, ha='right', va='center')
plt.text(w_mg, theta, r'$\theta$', fontsize=fontsize, ha='right', va='center')
plt.text(w_mg, w4, r'$w_4$', fontsize=fontsize, ha='right', va='center')

plt.text(t_center, t_bl, r'$(z_4>\theta)\cap(w_4>\theta)$', fontsize=fontsize, ha='center', va='baseline')
plt.axis('off')

plt.savefig('joint_density_function_rv_min_max_integration.pdf', bbox_inches='tight')
plt.show()

