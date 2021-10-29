import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.patches import Polygon
import matplotlib.colors as colors
from matplotlib import cm



__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)


# auxiliar function for plot ticks of equal length in x and y axis despite its scales.
# must be invoked after set the axes limits for example with xlim, ylim
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


def plot_pulse(x0, x1, h, lw, c):
    # horizontal lines
    plt.plot([x_min, x0], [0, 0], lw=lw, color=c)
    plt.plot([x0, x1], [h, h], lw=lw, color=c)
    plt.plot([x1, x_max], [0, 0], lw=lw, color=c)
    # vertical lines
    plt.plot([x0, x0], [0, h], lw=lw, color=c)
    plt.plot([x1, x1], [0, h], lw=lw, color=c)


# axis parameters
x_positive = 3

x_max = x_positive
x_min = -(x_positive - 1)
z_max = 2
z_min = -0.5

delta_ax = 0.3
x_ax_max = x_max + delta_ax
x_ax_min = x_min - delta_ax
z_ax_max = z_max
z_ax_min = z_min

h = 1
lw = 2

# colors
c1 = 'k'
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
c2 = scalarMap.to_rgba(1)
c3 = '#cccccc'

# axis labels parameters
bl = -0.45  # x labels baseline
rm = -0.1  # y labels right margin
fontsize = 14

# length of the ticks for all subplot (5 pixels)
display_length = 6  # in pixels

fig = plt.figure(1, figsize=(10, 8), frameon=False)
# SUBPLOT 1
ax = plt.subplot2grid((10, 10), (0, 0), rowspan=2, colspan=5)
plt.axis([x_ax_min, x_ax_max, z_ax_min, z_ax_max])

# axis arrows
plt.annotate("", xytext=(x_ax_min, 0), xycoords='data', xy=(x_ax_max, 0), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, z_ax_min), xycoords='data', xy=(0, z_ax_max), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plot_pulse(0, 1, h, lw, c1)
z = -0.7
plot_pulse(z-1, z, h, lw, c2)

# axis label
ax.text(x_ax_max, bl, r'$y$', fontsize=fontsize, ha='right', va='baseline', color='k')
ax.text(rm, bl, r'$0$', fontsize=fontsize, ha='right', va='baseline', color='k')
# pulses labels
ax.text(z-1, bl, r'$z-1$', fontsize=fontsize, ha='center', va='baseline', color=c2)
ax.text(z, bl, r'$z$', fontsize=fontsize, ha='center', va='baseline', color=c2)
ax.text(1, bl, r'$1$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(rm, bl, r'$0$', fontsize=fontsize, ha='right', va='baseline', color='k')
ax.text(rm, 1, r'$1$', fontsize=fontsize, ha='right', va='center', color='k')
# densities labels
ax.text(z-1/2, 1.3, r'$f_x(z-y)$', fontsize=fontsize, ha='center', va='baseline', color=c2)
ax.text(1, 1.3, r'$f_y(y)$', fontsize=fontsize, ha='center', va='baseline', color=c1)
# variable inequaltity
ax.text(x_max+0.1, z_max-0.4, r'$z<0$', fontsize=fontsize, ha='right', va='baseline', color=c1)
plt.axis('off')


# SUBPLOT 2
ax = plt.subplot2grid((10, 10), (2, 0), rowspan=2, colspan=5)
plt.axis([x_ax_min, x_ax_max, z_ax_min, z_ax_max])

# axis arrows
plt.annotate("", xytext=(x_ax_min, 0), xycoords='data', xy=(x_ax_max, 0), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, z_ax_min), xycoords='data', xy=(0, z_ax_max), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plot_pulse(0, 1, h, lw, c1)
z = 0.6
plot_pulse(z-1, z, h, lw, c2)

# axis label
ax.text(x_ax_max, bl, r'$y$', fontsize=fontsize, ha='right', va='baseline', color='k')
# pulses labels
ax.text(z-1, bl, r'$z-1$', fontsize=fontsize, ha='center', va='baseline', color=c2)
ax.text(z, bl, r'$z$', fontsize=fontsize, ha='center', va='baseline', color=c2)
ax.text(1, bl, r'$1$', fontsize=fontsize, ha='center', va='baseline', color='k')
# variable inequaltity
ax.text(x_max+0.1, z_max-0.4, r'$0<z<1$', fontsize=fontsize, ha='right', va='baseline', color=c1)
# filled rectangle
vertices = np.array([[0, 0], [z, 0], [z, 1], [0, 1]])
ax.add_patch(Polygon(vertices, facecolor=c3, edgecolor='none'))
plt.axis('off')

# SUBPLOT 3
ax = plt.subplot2grid((10, 10), (4, 0), rowspan=2, colspan=5)
plt.axis([x_ax_min, x_ax_max, z_ax_min, z_ax_max])

# axis arrows
plt.annotate("", xytext=(x_ax_min, 0), xycoords='data', xy=(x_ax_max, 0), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, z_ax_min), xycoords='data', xy=(0, z_ax_max), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plot_pulse(0, 1, h, lw, c1)
z = 1.4
plot_pulse(z-1, z, h, lw, c2)

# axis label
ax.text(x_ax_max, bl, r'$y$', fontsize=fontsize, ha='right', va='baseline', color='k')
ax.text(rm, bl, r'$0$', fontsize=fontsize, ha='right', va='baseline', color='k')
# pulses labels
ax.text(z-1, bl, r'$z-1$', fontsize=fontsize, ha='center', va='baseline', color=c2)
ax.text(z, bl, r'$z$', fontsize=fontsize, ha='center', va='baseline', color=c2)
ax.text(1, bl, r'$1$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(rm, 1, r'$1$', fontsize=fontsize, ha='right', va='center', color='k')
# variable inequaltity
ax.text(x_max+0.1, z_max-0.4, r'$1<z<2$', fontsize=fontsize, ha='right', va='baseline', color=c1)
# filled rectangle
vertices = np.array([[z-1, 0], [1, 0], [1, 1], [z-1, 1]])
ax.add_patch(Polygon(vertices, facecolor=c3, edgecolor='none'))
plt.axis('off')

# SUBPLOT 4
ax = plt.subplot2grid((10, 10), (6, 0), rowspan=2, colspan=5)
plt.axis([x_ax_min, x_ax_max, z_ax_min, z_ax_max])

# axis arrows
plt.annotate("", xytext=(x_ax_min, 0), xycoords='data', xy=(x_ax_max, 0), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, z_ax_min), xycoords='data', xy=(0, z_ax_max), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plot_pulse(0, 1, h, lw, c1)
z = 2.7
plot_pulse(z-1, z, h, lw, c2)

# axis label
ax.text(x_ax_max, bl, r'$y$', fontsize=fontsize, ha='right', va='baseline', color='k')
ax.text(rm, bl, r'$0$', fontsize=fontsize, ha='right', va='baseline', color='k')
# pulses labels
ax.text(z-1, bl, r'$z-1$', fontsize=fontsize, ha='center', va='baseline', color=c2)
ax.text(z, bl, r'$z$', fontsize=fontsize, ha='center', va='baseline', color=c2)
ax.text(1, bl, r'$1$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(rm, 1, r'$1$', fontsize=fontsize, ha='right', va='center', color='k')
# variable inequaltity
ax.text(x_max+0.1, z_max-0.4, r'$z>2$', fontsize=fontsize, ha='right', va='baseline', color=c1)
plt.axis('off')

# SUBPLOT 5
ax = plt.subplot2grid((10, 10), (8, 0), rowspan=2, colspan=5)
plt.axis([x_ax_min, x_ax_max, z_ax_min, z_ax_max])
# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(x_ax_min, 0), xycoords='data', xy=(x_ax_max, 0), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, z_ax_min), xycoords='data', xy=(0, z_ax_max), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot([x_min, 0], [0, 0], 'k', lw=2)
plt.plot([0, 1], [0, 1], 'k', lw=2)
plt.plot([1, 2], [1, 0], 'k', lw=2)
plt.plot([2, x_max], [0, 0], 'k', lw=2)

# ticks
plt.plot([1, 1], [0, ytl], 'k')
plt.plot([2, 2], [0, ytl], 'k')
plt.plot([0, xtl], [1, 1], 'k')

# axis label
ax.text(x_ax_max, bl, r'$z$', fontsize=fontsize, ha='right', va='baseline', color='k')
ax.text(rm, bl, r'$0$', fontsize=fontsize, ha='right', va='baseline', color='k')
# pulses labels
ax.text(2, bl, r'$2$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(1, bl, r'$1$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(rm, 1, r'$1$', fontsize=fontsize, ha='right', va='center', color='k')
# variable inequaltity
ax.text(0.15, z_max-0.25, r'$f_z(z)$', fontsize=fontsize, ha='left', va='baseline', color=c1)
plt.axis('off')

###############################################
###############################################
###############################################

# SUBPLOT 6
ax = plt.subplot2grid((10, 10), (0, 5), rowspan=2, colspan=5)
plt.axis([x_ax_min, x_ax_max, z_ax_min, z_ax_max])

# axis arrows
plt.annotate("", xytext=(x_ax_min, 0), xycoords='data', xy=(x_ax_max, 0), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, z_ax_min), xycoords='data', xy=(0, z_ax_max), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plot_pulse(0, 1, h, lw, c1)
z = -0.9
plot_pulse(z-1, z, h, lw, c2)

# axis label
ax.text(x_ax_max, bl, r'$x$', fontsize=fontsize, ha='right', va='baseline', color='k')
ax.text(rm, bl, r'$0$', fontsize=fontsize, ha='right', va='baseline', color='k')
# pulses labels
ax.text(z-1, bl, r'$w$', fontsize=fontsize, ha='center', va='baseline', color=c2)
ax.text(z, bl, r'$w+1$', fontsize=fontsize, ha='center', va='baseline', color=c2)
ax.text(1, bl, r'$1$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(rm, bl, r'$0$', fontsize=fontsize, ha='right', va='baseline', color='k')
ax.text(rm, 1, r'$1$', fontsize=fontsize, ha='right', va='center', color='k')
# densities labels
ax.text(z-1/2, 1.3, r'$f_y(x-w)$', fontsize=fontsize, ha='center', va='baseline', color=c2)
ax.text(1, 1.3, r'$f_x(x)$', fontsize=fontsize, ha='center', va='baseline', color=c1)
# variable inequaltity
ax.text(x_max+0.1, z_max-0.4, r'$w<-1$', fontsize=fontsize, ha='right', va='baseline', color=c1)
plt.axis('off')


# SUBPLOT 7
ax = plt.subplot2grid((10, 10), (2, 5), rowspan=2, colspan=5)
plt.axis([x_ax_min, x_ax_max, z_ax_min, z_ax_max])

# axis arrows
plt.annotate("", xytext=(x_ax_min, 0), xycoords='data', xy=(x_ax_max, 0), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, z_ax_min), xycoords='data', xy=(0, z_ax_max), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plot_pulse(0, 1, h, lw, c1)
z = 0.4
plot_pulse(z-1, z, h, lw, c2)

# axis label
ax.text(x_ax_max, bl, r'$x$', fontsize=fontsize, ha='right', va='baseline', color='k')
# pulses labels
ax.text(z-1, bl, r'$w$', fontsize=fontsize, ha='center', va='baseline', color=c2)
ax.text(z, bl, r'$w+1$', fontsize=fontsize, ha='center', va='baseline', color=c2)
ax.text(1, bl, r'$1$', fontsize=fontsize, ha='center', va='baseline', color='k')
# variable inequaltity
ax.text(x_max+0.1, z_max-0.4, r'$-1<w<0$', fontsize=fontsize, ha='right', va='baseline', color=c1)
# filled rectangle
vertices = np.array([[0, 0], [z, 0], [z, 1], [0, 1]])
ax.add_patch(Polygon(vertices, facecolor=c3, edgecolor='none'))
plt.axis('off')

# SUBPLOT 8
ax = plt.subplot2grid((10, 10), (4, 5), rowspan=2, colspan=5)
plt.axis([x_ax_min, x_ax_max, z_ax_min, z_ax_max])

# axis arrows
plt.annotate("", xytext=(x_ax_min, 0), xycoords='data', xy=(x_ax_max, 0), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, z_ax_min), xycoords='data', xy=(0, z_ax_max), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plot_pulse(0, 1, h, lw, c1)
z = 1.6
plot_pulse(z-1, z, h, lw, c2)

# axis label
ax.text(x_ax_max, bl, r'$x$', fontsize=fontsize, ha='right', va='baseline', color='k')
ax.text(rm, bl, r'$0$', fontsize=fontsize, ha='right', va='baseline', color='k')
# pulses labels
ax.text(z-1, bl, r'$w$', fontsize=fontsize, ha='center', va='baseline', color=c2)
ax.text(z, bl, r'$w+1$', fontsize=fontsize, ha='center', va='baseline', color=c2)
ax.text(1, bl, r'$1$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(rm, 1, r'$1$', fontsize=fontsize, ha='right', va='center', color='k')
# variable inequaltity
ax.text(x_max+0.1, z_max-0.4, r'$0<w<1$', fontsize=fontsize, ha='right', va='baseline', color=c1)
# filled rectangle
vertices = np.array([[z-1, 0], [1, 0], [1, 1], [z-1, 1]])
ax.add_patch(Polygon(vertices, facecolor=c3, edgecolor='none'))
plt.axis('off')

# SUBPLOT 9
ax = plt.subplot2grid((10, 10), (6, 5), rowspan=2, colspan=5)
plt.axis([x_ax_min, x_ax_max, z_ax_min, z_ax_max])

# axis arrows
plt.annotate("", xytext=(x_ax_min, 0), xycoords='data', xy=(x_ax_max, 0), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, z_ax_min), xycoords='data', xy=(0, z_ax_max), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plot_pulse(0, 1, h, lw, c1)
z = 2.5
plot_pulse(z-1, z, h, lw, c2)

# axis label
ax.text(x_ax_max, bl, r'$x$', fontsize=fontsize, ha='right', va='baseline', color='k')
ax.text(rm, bl, r'$0$', fontsize=fontsize, ha='right', va='baseline', color='k')
# pulses labels
ax.text(z-1, bl, r'$w$', fontsize=fontsize, ha='center', va='baseline', color=c2)
ax.text(z, bl, r'$w+1$', fontsize=fontsize, ha='center', va='baseline', color=c2)
ax.text(1, bl, r'$1$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(rm, 1, r'$1$', fontsize=fontsize, ha='right', va='center', color='k')
# variable inequaltity
ax.text(x_max+0.1, z_max-0.4, r'$w>1$', fontsize=fontsize, ha='right', va='baseline', color=c1)
plt.axis('off')

# SUBPLOT 10
ax = plt.subplot2grid((10, 10), (8, 5), rowspan=2, colspan=5)
plt.axis([x_ax_min, x_ax_max, z_ax_min, z_ax_max])
# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(x_ax_min, 0), xycoords='data', xy=(x_ax_max, 0), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, z_ax_min), xycoords='data', xy=(0, z_ax_max), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot([x_min, -1], [0, 0], 'k', lw=2)
plt.plot([-1, 0], [0, 1], 'k', lw=2)
plt.plot([0, 1], [1, 0], 'k', lw=2)
plt.plot([1, x_max], [0, 0], 'k', lw=2)

# ticks
plt.plot([1, 1], [0, ytl], 'k')
plt.plot([-1, -1], [0, ytl], 'k')

# axis label
ax.text(x_ax_max, bl, r'$w$', fontsize=fontsize, ha='right', va='baseline', color='k')
ax.text(rm, bl, r'$0$', fontsize=fontsize, ha='right', va='baseline', color='k')
# pulses labels
ax.text(-1, bl, r'$-1$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(1, bl, r'$1$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(rm, 1.1, r'$1$', fontsize=fontsize, ha='right', va='center', color='k')
# variable inequaltity
ax.text(0.15, z_max-0.25, r'$f_w(w)$', fontsize=fontsize, ha='left', va='baseline', color=c1)
plt.axis('off')


plt.savefig('joint_distribution_sum_sub_marginals_convolutions.pdf', bbox_inches='tight')
plt.show()




