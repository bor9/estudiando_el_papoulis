import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import matplotlib.colors as colors
from matplotlib.patches import Polygon

from matplotlib import cm
from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)
rc('mathtext', fontset='cm')


# auxiliar function for plot ticks of equal length in x and y axis despite its scales.
def convert_display_to_data_coordinates(transData, length=10):
    # create a transform which will take from display to data coordinates
    inv = transData.inverted()
    # transform from display coordinates to data coordinates in x axis
    data_coords = inv.transform([(0, 0), (length, 0)])
    # get the length of the segment in data units
    x_coord_len = data_coords[1, 0] - data_coords[0, 0]
    # transform from display coordinates to data coordinates in y axis
    data_coords = inv.transform([(0, 0), (0, length)])
    # get the length of the segment in data units
    y_coord_len = data_coords[1, 1] - data_coords[0, 1]
    return x_coord_len, y_coord_len

# f(x): normal density
mean = 0.4
variance = 1

# number of samples for the functions plot
N = 200

# range of x axis
xmin = -2
xmax = 3
# range of y axis
ymin = -0.15
ymax = 1.15

# normal distribution samples
x = np.linspace(xmin, xmax, N)
Fx = norm.cdf(x, loc=mean, scale=variance)

dx = -xmin/6
xk = np.arange(xmin+dx, xmax, dx)
Fk = norm.cdf(xk, loc=mean, scale=variance)

# extension to the left and right of the range of interest
delta_x = 0.2
delta_y = 0
x_ax_min = xmin - delta_x
x_ax_max = xmax + delta_x
y_ax_max = ymax + delta_y
y_ax_min = ymin - delta_y

# length of the ticks for all subplot in pixels
ticks_length = 5

x_bl = -0.08
y_rm = -0.1
font_size = 14

# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col10 = scalarMap.to_rgba(0)
col11 = scalarMap.to_rgba(0.2)
col20 = scalarMap.to_rgba(1)
col21 = scalarMap.to_rgba(0.8)
grey = scalarMap.to_rgba(0.5)

fig = plt.figure(0, figsize=(10, 4), frameon=False)
# SUBPLOT 1
ax = plt.subplot2grid((1, 10), (0, 0), rowspan=1, colspan=5)
plt.axis([x_ax_min, x_ax_max, y_ax_min, y_ax_max])

# horizontal and vertical ticks length
htl, vtl = convert_display_to_data_coordinates(ax.transData, length=ticks_length)

# axis arrows
plt.annotate("", xytext=(x_ax_min, 0), xycoords='data', xy=(x_ax_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=5, headlength=7, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, y_ax_min), xycoords='data', xy=(0, y_ax_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=5, headlength=7, facecolor='black', shrink=0.002))

# axis labels
plt.text(x_ax_max, x_bl, r'$x$', fontsize=font_size, ha='right', va='baseline')
plt.text(y_rm, y_ax_max, r'$F(x)$', fontsize=font_size, ha='right', va='center')


ax.fill_between(x[x >= 0], Fx[x >= 0], 1, color=grey)
ax.fill_between(x[x <= 0], 0, Fx[x <= 0], color=grey)

plt.plot(x, Fx, 'k', lw=2)

# ticks
xk_pos = xk[xk >= 0]
Fk_pos = Fk[xk >= 0]
xk_neg = xk[xk <= 0]
Fk_neg = Fk[xk <= 0]

plt.plot([xk, xk], [0, vtl], 'k')

plt.plot([np.zeros(xk_pos.shape), xk_pos], [Fk_pos, Fk_pos], color=col10)
plt.plot([np.zeros(xk_neg.size-1), xk_neg[:-1]], [Fk_neg[1:], Fk_neg[1:]], color=col20)
plt.plot([xk_pos[:-1], xk_pos[:-1]], [Fk_pos[:-1], Fk_pos[1:]], color=col10)
plt.plot([xk_neg[:-1], xk_neg[:-1]], [Fk_neg[:-1], Fk_neg[1:]], color=col20)

i_pos = 7
plt.plot([xk[i_pos], xk[i_pos]], [0, Fk[i_pos]], 'k--', dashes=(5, 3))
plt.plot([xk[i_pos+1], xk[i_pos+1]], [0, Fk[i_pos+1]], 'k--', dashes=(5, 3))
# colored stripe
vertices = np.array([[0, Fk[i_pos]], [xk[i_pos], Fk[i_pos]], [xk[i_pos], Fk[i_pos+1]], [0, Fk[i_pos+1]]])
ax.add_patch(Polygon(vertices, facecolor=col11, edgecolor='none'))
i_neg = 2
plt.plot([xk[i_neg], xk[i_neg]], [0, Fk[i_neg]], 'k--', dashes=(5, 3))
plt.plot([xk[i_neg+1], xk[i_neg+1]], [0, Fk[i_neg+1]], 'k--', dashes=(5, 3))
# colored stripe
vertices = np.array([[0, Fk[i_neg]], [xk[i_neg], Fk[i_neg]], [xk[i_neg], Fk[i_neg+1]], [0, Fk[i_neg+1]]])
ax.add_patch(Polygon(vertices, facecolor=col21, edgecolor='none'))

# labels
plt.text(xk[i_pos], x_bl, r'$x_k$', fontsize=font_size, ha='center', va='baseline')
plt.text(xk[i_pos+1]+2*dx/3, x_bl, r'$x_{k+1}$', fontsize=font_size, ha='center', va='baseline')
plt.text(xk[i_neg], x_bl, r"$x_{k'}$", fontsize=font_size, ha='center', va='baseline')
plt.text(xk[i_neg+1]+2*dx/3, x_bl, r"$x_{k'+1}$", fontsize=font_size, ha='center', va='baseline')

plt.text(y_rm, Fk[i_pos], r'$F(x_k)$', fontsize=font_size, ha='right', va='center')
plt.text(y_rm, Fk[i_pos+1], r'$F(x_{k+1})$', fontsize=font_size, ha='right', va='center')
plt.text(y_rm, 1, r'$1$', fontsize=font_size, ha='right', va='center')

# 1/n
im = 10
yd = 0.1
plt.plot([xk[im], xk[im]], [yd-vtl, yd+vtl], 'k')
plt.plot([xk[im+1], xk[im+1]], [yd-vtl, yd+vtl], 'k')
plt.plot([xk[im], xk[im+1]], [yd, yd], 'k')
plt.text((xk[im]+xk[im+1])/2, yd+2*vtl, r'$\Delta x$', fontsize=font_size, ha='center', va='baseline')

plt.axis('off')


ax = plt.subplot2grid((1, 10), (0, 5), rowspan=1, colspan=5)
plt.axis([x_ax_min, x_ax_max, y_ax_min, y_ax_max])

# axis arrows
plt.annotate("", xytext=(x_ax_min, 0), xycoords='data', xy=(x_ax_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=5, headlength=7, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, y_ax_min), xycoords='data', xy=(0, y_ax_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=5, headlength=7, facecolor='black', shrink=0.002))

# axis labels
plt.text(x_ax_max, x_bl, r'$x$', fontsize=font_size, ha='right', va='baseline')
plt.text(y_rm, y_ax_max, r'$F(x)$', fontsize=font_size, ha='right', va='center')
plt.text(y_rm, 1, r'$1$', fontsize=font_size, ha='right', va='center')
plt.text(y_rm, x_bl, r'$0$', fontsize=font_size, ha='right', va='baseline')

plt.text(xmin, x_bl, r'$A$', fontsize=font_size, ha='center', va='baseline')
plt.text(y_rm, norm.cdf(0, loc=mean, scale=variance), r'$B$', fontsize=font_size, ha='right', va='center')
plt.text(xmax, 0.98, r'$D$', fontsize=font_size, ha='center', va='top')


ax.fill_between(x[x >= 0], Fx[x >= 0], 1, color=col11)
ax.fill_between(x[x <= 0], 0, Fx[x <= 0], color=col21)

plt.plot(x, Fx, 'k', lw=2)
plt.plot([0, xmax], [1, 1], 'k')

x1 = 0.6
y1 = 0.8
dy1 = 0.065
Fx1 = norm.cdf(x1, loc=mean, scale=variance)
plt.annotate(s='', xytext=(x1, y1+dy1), xy=(x1, 1),
             arrowprops=dict(arrowstyle='->, head_width=0.15, head_length=0.4', facecolor='black',
                             shrinkA=0, shrinkB=0))
plt.annotate(s='', xytext=(x1, y1-dy1), xy=(x1, Fx1),
             arrowprops=dict(arrowstyle='->, head_width=0.15, head_length=0.4', facecolor='black',
                             shrinkA=0, shrinkB=0))
plt.text(x1, y1, r'$1-F(x)$', fontsize=font_size, ha='center', va='center')

plt.annotate(s='', xytext=(x1, 0), xy=(x1, Fx1),
             arrowprops=dict(arrowstyle='<->, head_width=0.15, head_length=0.4', facecolor='black',
                             shrinkA=0, shrinkB=0))
plt.text(x1, Fx1/2, r'$F(x)$', fontsize=font_size, ha='center', va='center',
         bbox=dict(facecolor='w', edgecolor='none', pad=5))

plt.axis('off')

plt.savefig('mean_and_distribution.pdf', bbox_inches='tight')
plt.show()
