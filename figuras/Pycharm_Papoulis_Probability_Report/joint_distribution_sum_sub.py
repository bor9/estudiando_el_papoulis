import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from matplotlib import rc
import matplotlib.colors as colors
from matplotlib import cm
import math


__author__ = 'ernesto'


def point_in_line_given_distance(m, n, d, x1):
    """
    :param m:
    :param n:
    :param d:
    :param x1:
    :return:
    """
    y1 = m * x1 + n
    A = m ** 2 + 1
    B = 2 * (m*n - m*y1 - x1)
    C = y1 ** 2 - d ** 2 + x1 ** 2 - 2 * n * y1 + n ** 2
    x2 = np.amin(np.roots([A, B, C]))
    y2 = m * x2 + n
    return x2, y2


# if use latex or mathtext
rc('text', usetex=False)

# axis limits
w_min = -2.5
w_max = 2.5
z_min = -0.8
z_max = 2.8

delta_ax = 0.3
z_ax_max = z_max + delta_ax
z_ax_min = z_min - delta_ax
w_ax_max = w_max + delta_ax
w_ax_min = w_min - delta_ax
# y tick margin
ytm = 0.6
# font size
font_size = 16
# colors
c = np.linspace(0.8, 0.4, 8)

c2 = np.linspace(0, 1, 8)
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)

fig = plt.figure(1, figsize=(8, 4), frameon=False)
#
# REGION 1
#
ax = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)
plt.axis([z_ax_min, z_ax_max, w_ax_min, w_ax_max])
ax.set_aspect('equal', adjustable='box')


# axis arrows
plt.annotate("", xytext=(z_ax_min, 0), xycoords='data', xy=(z_ax_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, w_ax_min), xycoords='data', xy=(0, w_ax_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# axis labels
plt.text(z_ax_max, -0.3, r'$z$', fontsize=font_size, ha='center', va='baseline')
plt.text(-0.1, w_ax_max, r'$w$', fontsize=font_size, ha='right', va='center')

# region 1
# region limit (w=z)
plt.plot([z_min, w_max], [z_min, w_max], 'k')
# filled region
vertices = np.array([[z_min, z_min], [w_max, w_max], [z_max, w_max], [z_max, w_min], [z_min, w_min]])
#ax.add_patch(Polygon(vertices, facecolor=c[0]*np.ones((3,)), alpha=0.4, edgecolor='none'))
ax.add_patch(Polygon(vertices, facecolor=scalarMap.to_rgba(c2[0]), alpha=0.4, edgecolor='none'))


# region 2
# region limit (w=-z)
plt.plot([z_min, w_max], [-z_min, -w_max], 'k')
# filled region
vertices = np.array([[z_min, -z_min], [z_min, w_max], [z_max, w_max], [z_max, w_min], [-w_min, w_min]])
#ax.add_patch(Polygon(vertices, facecolor=c[1]*np.ones((3,)), alpha=0.4, edgecolor='none'))
ax.add_patch(Polygon(vertices, facecolor=scalarMap.to_rgba(c2[1]), alpha=0.4, edgecolor='none'))

# region 3
# region limit (z+w=2)
plt.plot([2-w_max, z_max], [w_max, 2-z_max], 'k')
# filled region (x<y)
vertices = np.array([[2-w_max, w_max], [z_min, w_max], [z_min, w_min], [z_max, w_min], [z_max, 2-z_max]])
#ax.add_patch(Polygon(vertices, facecolor=c[2]*np.ones((3,)), alpha=0.4, edgecolor='none'))
ax.add_patch(Polygon(vertices, facecolor=scalarMap.to_rgba(c2[2]), alpha=0.8, edgecolor='none'))

# region 4
# region limit (z-w=2)
plt.plot([2+w_min, z_max], [w_min, z_max-2], 'r')
# filled region (x<y)
vertices = np.array([[2+w_min, w_min], [z_min, w_min], [z_min, w_max], [z_max, w_max], [z_max, z_max-2]])
#ax.add_patch(Polygon(vertices, facecolor=c[3]*np.ones((3,)), alpha=0.4, edgecolor='none'))
ax.add_patch(Polygon(vertices, facecolor=scalarMap.to_rgba(c2[3]), alpha=0.4, edgecolor='none'))

# region 5
# region limit (w=1)
plt.plot([z_min, z_max], [1, 1], 'k')

x1 = 2+w_min
y1 = w_min
x2 = z_max
y2 = z_max-2

ticks_sep = 0.1
ticks_len = 0.2
ticks_angle = math.pi/3
line_len = math.sqrt((x2-x1)**2 + (y2-y1)**2)
ticks_len_line = ticks_len * math.cos(ticks_angle)
nticks = math.ceil((line_len-ticks_len_line)/ticks_sep)

# line equation
m = (y2-y1) / (x2-x1)
n = -m * x1 + y1
alpha = math.atan(m)
xi = x1 + math.cos(alpha) * (ticks_len_line + ticks_sep * np.arange(nticks))

plt.plot(xi, m*xi+n, 'k.')

mt = math.tan(alpha - ticks_angle)
print(mt)

for x in xi:
    nt = -mt * x + m * x + n
    xk, yk = point_in_line_given_distance(mt, -mt*x+m*x+n, ticks_len, x)
    plt.plot([x, xk], [mt*x + nt, mt*xk + nt], 'k', lw=0.5)

# region 6
# region limit (w=-1)
plt.plot([z_min, z_max], [-1, -1], 'k')
# filled region (x<y)


#
#
# # labels
# plt.text(t_min/2, t_max/2, r'$x\leq z$', fontsize=font_size, ha='center', va='center')
# plt.text(z, t_max, r'$x=z$', fontsize=font_size, ha='center', va='bottom')
#
# plt.text(t_max-3, t_max-4, r'$x=y$', fontsize=font_size, ha='center', va='bottom', rotation=45)
# plt.text((z+t_max)/2, t_min/2, r'$x>y$', fontsize=font_size, ha='center', va='center')
#
# plt.text(z+0.4, z-0.7, r'$(z,\,z)$', fontsize=font_size, ha='left', va='center')
# plt.plot(z, z, 'k.', markersize=8)
#
# plt.text((t_min+t_max)/2, ax_min-3.5, r'$P\{\mathbf{x}\leq z,\,\mathbf{x}>\mathbf{y}\}$',
#          fontsize=font_size, ha='center', va='baseline')
#
# plt.axis('off')
#
# #
# # REGION 2
# #
# ax = plt.subplot2grid((2, 4), (0, 2), rowspan=2, colspan=2)
# plt.axis([ax_min, ax_max, ax_min, ax_max])
# ax.set_aspect('equal', adjustable='box')
#
#
# # axis arrows
# plt.annotate("", xytext=(ax_min, 0), xycoords='data', xy=(ax_max, 0), textcoords='data',
#              arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
# plt.annotate("", xytext=(0, ax_min), xycoords='data', xy=(0, ax_max), textcoords='data',
#              arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
#
# # axis labels
# plt.text(ax_max, -1.8, r'$x$', fontsize=font_size, ha='center', va='baseline')
# plt.text(-0.6, ax_max, r'$y$', fontsize=font_size, ha='right', va='center')
#
# # region 1
# # region limit (x=y)
# plt.plot([t_min, t_max], [t_min, t_max], 'k')
# # filled region (x<y)
# vertices = np.array([[t_max, t_max], [t_min, t_min], [t_min, t_max]])
# ax.add_patch(Polygon(vertices, facecolor='#0343df', alpha=0.4, edgecolor='none'))
#
# # region 2
# # region limit (x=z)
# plt.plot([t_min, t_max], [z, z], 'k')
# # filled region (x<y)
# vertices = np.array([[t_max, t_min], [t_max, z], [t_min, z], [t_min, t_min]])
# ax.add_patch(Polygon(vertices, facecolor='#ff000d', alpha=0.4, edgecolor='none'))
#
#
# # labels
# plt.text(t_min/2, (z+t_max)/2, r'$x\leq y$', fontsize=font_size, ha='center', va='center')
# plt.text(t_max, t_max/2, r'$y=z$', fontsize=font_size, ha='center', va='center')
#
# plt.text(t_max, t_max-4, r'$x=y$', fontsize=font_size, ha='center', va='bottom', rotation=45)
# plt.text(t_max/2, t_min/2, r'$y\leq z$', fontsize=font_size, ha='center', va='center')
#
# plt.text(z-0.4, z-1.6, r'$(z,\,z)$', fontsize=font_size, ha='left', va='center')
# plt.plot(z, z, 'k.', markersize=8)
#
# plt.text((t_min+t_max)/2, ax_min-3.5, r'$P\{\mathbf{y}\leq z,\,\mathbf{x}\leq \mathbf{y}\}$',
#          fontsize=font_size, ha='center', va='baseline')
#
plt.axis('off')
plt.savefig('joint_distribution_sum_sub.pdf', bbox_inches='tight')

plt.show()

