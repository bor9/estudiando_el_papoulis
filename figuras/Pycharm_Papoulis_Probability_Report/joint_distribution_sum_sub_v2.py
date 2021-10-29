import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import matplotlib.colors as colors
from matplotlib import cm
import math


__author__ = 'ernesto'


def point_in_line_given_distance(m, n, d, x1, left_position=True):
    """
    Computes the coordinates of a point (x2, y2) of distance d of another point (x1, y1), both over a line
    y = mx + n
    :param m: line slope
    :param n: line cut with y axes
    :param d: distance
    :param x1: reference point over the line
    :param left_position: there are two solutions. select the returned solution
    :return: coordinates of the point
    """
    y1 = m * x1 + n
    A = m ** 2 + 1
    B = 2 * (m*n - m*y1 - x1)
    C = y1 ** 2 - d ** 2 + x1 ** 2 - 2 * n * y1 + n ** 2
    if left_position:
        x2 = np.amin(np.roots([A, B, C]))
    else:
        x2 = np.amax(np.roots([A, B, C]))
    y2 = m * x2 + n
    return x2, y2


def half_plane_ticks(x1, y1, x2, y2, left_position=True):
    # length of the line
    line_len = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # length of the proyection of ticks over the line
    ticks_len_x = ticks_len * math.cos(ticks_angle)
    # number of ticks in the line
    nticks = math.ceil((line_len - ticks_len_x) / ticks_sep)

    # line equation (y = mx + n)
    m = (y2 - y1) / (x2 - x1)
    n = -m * x1 + y1
    # x coordinates of beginning of ticks
    alpha = math.atan(m)
    xi = x1 + math.cos(alpha) * (ticks_len_x + ticks_sep * np.arange(nticks))

    # parallel line for the end of ticks
    # line perpendicular to the line for (x1, y1)
    mper = -1 / m
    nper = -mper * x1 + y1
    ticks_len_y = ticks_len * math.sin(ticks_angle)
    xp, yp = point_in_line_given_distance(mper, nper, ticks_len_y, x1, left_position=left_position)
    mt = m
    nt = -mt * xp + yp
    # x coordinates of the end of ticks
    xf = xp + + math.cos(alpha) * (ticks_sep * np.arange(nticks))
    return xi, xf, m*xi + n, mt*xf + nt


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
c = np.linspace(0, 1, 4)
print(c)
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)

# half-plane ticks parameters
ticks_sep = 0.1
ticks_len = 0.2
ticks_angle = math.pi/3

fig = plt.figure(1, figsize=(8, 4), frameon=False)
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
plt.plot([z_min, w_max], [z_min, w_max], color=scalarMap.to_rgba(c[0]))
xi, xf, yi, yf = half_plane_ticks(z_min, z_min, w_max, w_max, left_position=False)
plt.plot([xi, xf], [yi, yf], color=scalarMap.to_rgba(c[0]), lw=0.5)
plt.text(z_max-0.4, w_max-1.1, r'$w<z$', fontsize=font_size, ha='center', va='bottom', rotation=45,
         color=scalarMap.to_rgba(c[0]))


# region 2
# region limit (w=-z)
plt.plot([z_min, w_max], [-z_min, -w_max], color=scalarMap.to_rgba(c[1]))
xi, xf, yi, yf = half_plane_ticks(z_min, -z_min, w_max, -w_max, left_position=False)
plt.plot([xi, xf], [yi, yf], color=scalarMap.to_rgba(c[1]), lw=0.5)

# region 3
# region limit (z+w=2)
plt.plot([2-w_max, z_max], [w_max, 2-z_max], color=scalarMap.to_rgba(c[2]))
xi, xf, yi, yf = half_plane_ticks(2-w_max, w_max, z_max, 2-z_max, left_position=True)
plt.plot([xi, xf], [yi, yf], color=scalarMap.to_rgba(c[2]), lw=0.5)

# region 4
# region limit (z-w=2)
plt.plot([2+w_min, z_max], [w_min, z_max-2], color=scalarMap.to_rgba(c[3]))
xi, xf, yi, yf = half_plane_ticks(2+w_min, w_min, z_max, z_max-2, left_position=True)
plt.plot([xi, xf], [yi, yf], color=scalarMap.to_rgba(c[3]), lw=0.5)
plt.text(z_max-0.4, 0.3, r'$w>z-2$', fontsize=font_size, ha='center', va='bottom', rotation=45,
         color=scalarMap.to_rgba(c[3]))
plt.text(z_min+0.6, w_min+0.2, r'$w>z-2$', fontsize=font_size, ha='center', va='bottom', rotation=45,
         color=scalarMap.to_rgba(c[3]), bbox=dict(facecolor='w', edgecolor='none', pad=0))


plt.axis('off')
plt.savefig('joint_distribution_sum_sub.pdf', bbox_inches='tight')

plt.show()

