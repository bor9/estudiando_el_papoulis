import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import rc
import matplotlib.colors as colors
from matplotlib import cm

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


#####################################
# PARAMETERS - This can be modified #
#####################################

# tirangular density: (2x)/a^2, 0 < x < a
a = 2

#####################
# END OF PARAMETERS #
#####################

# axis parameters
xmin = -0.6
xmax = a + 0.6
ymin = -0.35
ymax = 2/a + 0.35

dx = 0.25
xmin_ax = xmin - dx
xmax_ax = xmax + dx
ymin_ax = ymin
ymax_ax = ymax

# abscissa values
x = np.linspace(xmin, xmax, 500)

# DENSITY AND DISTRIBUTION CONSTRUCTION
# index of 0 and a values in vector x
idx_0 = np.argmax(x > 0)
idx_a = np.argmax(x > a) - 1
# density function
fx = np.zeros(x.shape)
fx[idx_0:idx_a] = 2 * x[idx_0:idx_a] / a ** 2
# distribution function
Fx = np.zeros(x.shape)
Fx[idx_0:idx_a] = (x[idx_0:idx_a]/a) ** 2
Fx[idx_a:] = 1


# vertical tick margin
vtm = -0.15
# horizontal tick margin
htm = -0.08
# font size
fontsize = 14
# length of the ticks for all subplot (7 pixels)
display_length = 7  # in pixels

# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col0 = scalarMap.to_rgba(0.2)
col1 = scalarMap.to_rgba(0.8)
blue = scalarMap.to_rgba(0)

##############
#    PLOT    #
##############
fig = plt.figure(0, figsize=(10, 3), frameon=False)

# DENSITY PLOT #
ax = plt.subplot2grid((1, 6), (0, 0), rowspan=1, colspan=3)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
htl, vtl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# density plot
plt.plot(x, fx, 'k', linewidth=2.5)
# ticks
plt.plot([a, a], [0, vtl], 'k')
plt.plot([2*a/3, 2*a/3], [0, vtl], 'k')
plt.plot([a/math.sqrt(2), a/math.sqrt(2)], [0, vtl], 'k')
plt.plot([0, htl], [2/a, 2/a], 'k')
#
plt.plot([a/math.sqrt(2), a/math.sqrt(2)], [0, math.sqrt(2)/a], 'k', linewidth=0.8)

# ticks labels
# xlabels
plt.text(xmax_ax, vtm, '$x$', fontsize=fontsize, ha='right', va='baseline')
plt.text(a, vtm, '$a$', fontsize=fontsize, ha='center', va='baseline')
plt.text(htm, vtm, '$0$', fontsize=fontsize, ha='right', va='baseline')
# ylabels
plt.text(htm, 1, r'$2/a$', fontsize=fontsize, ha='right', va='center')
plt.text(htm, ymax_ax, r'$f_x(x)$', fontsize=fontsize, ha='right', va='center')

td = 0.2
plt.annotate(r'$\eta_x=2a/3$', xytext=(2*a/3-td, -0.35), xycoords='data', xy=(2*a/3, 0),
             textcoords='data', fontsize=fontsize, va="baseline", ha="right",
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.4", facecolor='black', relpos=(1, 1),
                             patchA=None, patchB=None, shrinkA=4, shrinkB=1))

plt.annotate(r'$x_{0.5}=a/\sqrt{2}$', xytext=(a/math.sqrt(2)+td, -0.35), xycoords='data', xy=(a/math.sqrt(2), 0),
             textcoords='data', fontsize=fontsize, va="baseline", ha="left",
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.4", facecolor='black', relpos=(0, 1),
                             patchA=None, patchB=None, shrinkA=1, shrinkB=1))


idx_median = np.argmax(x > a/math.sqrt(2))
ax.fill_between(x[idx_0: idx_median], 0, fx[idx_0: idx_median], color=col0)
ax.fill_between(x[idx_median: idx_a], 0, fx[idx_median: idx_a], color=col1)


plt.axis('off')

# DISTRIBUTION PLOT #
ax = plt.subplot2grid((1, 6), (0, 3), rowspan=1, colspan=3)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
htl, vtl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# distribution plot
plt.plot(x, Fx, 'k', linewidth=2.5)
# ticks
plt.plot([a, a], [0, vtl], 'k')
plt.plot([2*a/3, 2*a/3], [0, vtl], 'k')
plt.plot([a/math.sqrt(2), a/math.sqrt(2)], [0, vtl], 'k')
plt.plot([0, htl], [1, 1], 'k')
plt.plot([0, htl], [1/2, 1/2], 'k')
#
plt.plot([a/math.sqrt(2), a/math.sqrt(2)], [0, 1/2], 'k', linewidth=0.8)
plt.plot([0, a/math.sqrt(2)], [1/2, 1/2], 'k', linewidth=0.8)

# ticks labels
# xlabels
plt.text(xmax_ax, vtm, '$x$', fontsize=fontsize, ha='right', va='baseline')
plt.text(a, vtm, '$a$', fontsize=fontsize, ha='center', va='baseline')
plt.text(htm, vtm, '$0$', fontsize=fontsize, ha='right', va='baseline')
# ylabels
plt.text(htm, 1, r'$1$', fontsize=fontsize, ha='right', va='center')
plt.text(htm, 1/2, r'$1/2$', fontsize=fontsize, ha='right', va='center')
plt.text(htm, ymax_ax, r'$F_x(x)$', fontsize=fontsize, ha='right', va='center')

td = 0.2
plt.annotate(r'$\eta_x=2a/3$', xytext=(2*a/3-td, -0.35), xycoords='data', xy=(2*a/3, 0),
             textcoords='data', fontsize=fontsize, va="baseline", ha="right",
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.4", facecolor='black', relpos=(1, 1),
                             patchA=None, patchB=None, shrinkA=4, shrinkB=1))

plt.annotate(r'$x_{0.5}=a/\sqrt{2}$', xytext=(a/math.sqrt(2)+td, -0.35), xycoords='data', xy=(a/math.sqrt(2), 0),
             textcoords='data', fontsize=fontsize, va="baseline", ha="left",
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.4", facecolor='black', relpos=(0, 1),
                             patchA=None, patchB=None, shrinkA=1, shrinkB=1))

plt.axis('off')

# plt.plot(x, Fx, 'k', linewidth=2)



# save as pdf
plt.savefig('mean_and_median_example.pdf', bbox_inches='tight')
plt.show()


