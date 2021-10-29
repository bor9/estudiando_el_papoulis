import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.patches import Polygon

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)

z1 = 0.7
z2 = 1.5
dz = 0.02
w1 = 0.5
w2 = -0.8
dw = dz

# axis limits
w_min = -1.2
w_max = 1.2
z_min = -0.2
z_max = 2.5

delta_ax = 0.2
z_ax_max = z_max + delta_ax
z_ax_min = z_min - delta_ax
w_ax_max = w_max + delta_ax
w_ax_min = w_min - delta_ax
# font size
font_size = 16

# ticks
# ticks length
tl = 0.1
# z ticks labels baseline
bl = -0.2
# colors
lgray = "#dddddd"  # ligth gray
dgray = "k"  # dark gray

fig = plt.figure(1, figsize=(10, 4), frameon=False)
# SUBPLOT 1
ax = plt.subplot2grid((1, 10), (0, 0), rowspan=1, colspan=5)
plt.axis([z_ax_min, z_ax_max, w_ax_min, w_ax_max])
ax.set_aspect('equal', adjustable='box')

# axis arrows
plt.annotate("", xytext=(z_ax_min, 0), xycoords='data', xy=(z_ax_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, w_ax_min), xycoords='data', xy=(0, w_ax_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# axis labels
plt.text(-0.06, w_ax_max, r'$w$', fontsize=font_size, ha='right', va='center')
plt.text(z_ax_max, bl, r'$z$', fontsize=font_size, ha='center', va='baseline')

# region limit (w=z)
plt.plot([0, 1], [0, 1], 'k')
# region limit (w=-z)
plt.plot([0, 1], [0, -1], 'k')
# region limit (z+w=2)
plt.plot([1, 2], [1, 0], 'k')
# region limit (z-w=2)
plt.plot([1, 2], [-1, 0], 'k')

# yticks and labels
plt.plot([0, tl], [1, 1], 'k')
plt.plot([0, tl], [-1, -1], 'k')
plt.text(-0.06, 1, r'$1$', fontsize=font_size, ha='right', va='center')
plt.text(-0.06, -1, r'$-1$', fontsize=font_size, ha='right', va='center')
# xticks and labels
plt.plot([2, 2], [0, tl], 'k')
plt.plot([1, 1], [0, tl], 'k')
plt.text(2, bl, r'$2$', fontsize=font_size, ha='center', va='baseline')
plt.text(1, bl, r'$1$', fontsize=font_size, ha='center', va='baseline')

# density region
vertices = np.array([[0, 0], [1, 1], [2, 0], [1, -1]])
ax.add_patch(Polygon(vertices, facecolor=lgray, edgecolor='none'))

# integration line
plt.plot([z1, z1], [-z1, z1], 'k')
plt.plot([z1+dz, z1+dz], [-z1-dz, z1+dz], 'k')
vertices = np.array([[z1, z1], [z1+dz, z1+dz], [z1+dz, -z1-dz], [z1, -z1]])
ax.add_patch(Polygon(vertices, facecolor=dgray, edgecolor='none'))
plt.text(z1, z1+dz/2+0.05, r'$w=z$', fontsize=font_size, ha='right', va='center')
plt.text(z1, -z1-0.1, r'$w=-z$', fontsize=font_size, ha='right', va='center')

plt.plot([z2, z2], [z2-2, -z2+2], 'k')
plt.plot([z2+dz, z2+dz], [z2+dz-2, -z2-dz+2], 'k')
vertices = np.array([[z2, z2-2], [z2+dz, z2+dz-2], [z2+dz, -z2-dz+2], [z2, -z2+2]])
ax.add_patch(Polygon(vertices, facecolor=dgray, edgecolor='none'))
plt.text(z2+0.05, -z2+2+0.05, r'$w=-z+2$', fontsize=font_size, ha='left', va='center')
plt.text(z2+0.05, z2-2-0.05, r'$w=z-2$', fontsize=font_size, ha='left', va='center')

plt.axis('off')

# SUBPLOT 2
ax = plt.subplot2grid((1, 10), (0, 5), rowspan=1, colspan=5)
plt.axis([z_ax_min, z_ax_max, w_ax_min, w_ax_max])
ax.set_aspect('equal', adjustable='box')

# axis arrows
plt.annotate("", xytext=(z_ax_min, 0), xycoords='data', xy=(z_ax_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, w_ax_min), xycoords='data', xy=(0, w_ax_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# axis labels
plt.text(-0.06, w_ax_max, r'$w$', fontsize=font_size, ha='right', va='center')
plt.text(z_ax_max, bl, r'$z$', fontsize=font_size, ha='center', va='baseline')

# region limit (w=z)
plt.plot([0, 1], [0, 1], 'k')
# region limit (w=-z)
plt.plot([0, 1], [0, -1], 'k')
# region limit (z+w=2)
plt.plot([1, 2], [1, 0], 'k')
# region limit (z-w=2)
plt.plot([1, 2], [-1, 0], 'k')

# yticks and labels
plt.plot([0, tl], [1, 1], 'k')
plt.plot([0, tl], [-1, -1], 'k')
plt.text(-0.06, 1, r'$1$', fontsize=font_size, ha='right', va='center')
plt.text(-0.06, -1, r'$-1$', fontsize=font_size, ha='right', va='center')
# xticks and labels
plt.plot([2, 2], [0, tl], 'k')
plt.plot([1, 1], [0, tl], 'k')
plt.text(2, bl, r'$2$', fontsize=font_size, ha='center', va='baseline')
plt.text(1, bl, r'$1$', fontsize=font_size, ha='center', va='baseline')

# density region
vertices = np.array([[0, 0], [1, 1], [2, 0], [1, -1]])
ax.add_patch(Polygon(vertices, facecolor=lgray, edgecolor='none'))

# integration line
plt.plot([w1, -w1+2], [w1, w1], 'k')
plt.plot([w1+dw, -w1-dw+2], [w1+dw, w1+dw], 'k')
vertices = np.array([[w1, w1], [-w1+2, w1], [-w1-dw+2, w1+dw], [w1+dw, w1+dw]])
ax.add_patch(Polygon(vertices, facecolor=dgray, edgecolor='none'))
plt.text(w1-0.01, w1+0.01, r'$z=w$', fontsize=font_size, ha='right', va='baseline')
plt.text(2-w1+0.03, w1+0.01, r'$z=-w+2$', fontsize=font_size, ha='left', va='baseline')

plt.plot([-w2, w2+2], [w2, w2], 'k')
plt.plot([-w2-dw, w2+dw+2], [w2+dw, w2+dw], 'k')
vertices = np.array([[-w2, w2], [-w2-dw, w2+dw], [w2+dw+2, w2+dw], [w2+2, w2]])
ax.add_patch(Polygon(vertices, facecolor=dgray, edgecolor='none'))
plt.text((-w2-dw)/2, w2-0.05, r'$z=-w$', fontsize=font_size, ha='center', va='baseline')
plt.text(w2+2+0.08, w2-0.05, r'$z=w+2$', fontsize=font_size, ha='left', va='baseline')

plt.axis('off')
plt.savefig('joint_distribution_sum_sub_marginals.pdf', bbox_inches='tight')

plt.show()
