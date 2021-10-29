import matplotlib.pyplot as plt
import numpy as np
import math

from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=True)
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

# sinusoid period and phase (samples)
N = 50
# decaying sinusoid amplitude
a = 1
# decaying rate
tau = 50

# number of periods
nN = 6

# first and last sample number
nmin = 0
nmax = N * nN

# extension to the left and right of the range of interest
delta_n = 10

# x and y axis maximum value
xmin = nmin
xmax = nmax + delta_n
ymax = 2
ymin = -ymax

#####################
# END OF PARAMETERS #
#####################

# samples
n = np.arange(nmin, nmax)

# sinusoid angular frequency (rad)
theta = 2 * math.pi / N

# decaying sinusoid
xn = a * (1 + np.exp(-n/tau)) * np.sin(2 * math.pi / N * n)
# sup and inf computation
sup_xn = np.zeros(xn.shape)
inf_xn = np.zeros(xn.shape)

for k in n:
    sup_xn[k] = np.maximum(1, np.amax(xn[k:nmax]))
    inf_xn[k] = np.minimum(-1, np.amin(xn[k:nmax]))

# horizontal label margin
hlm = -0.16
# horizontal label margin
vlm = 4
# font size
fontsize = 14
# dashes length/space
dashed = (4, 4)
# length of the ticks for all subplot (7 pixels)
display_length = 6  # in pixels

fig = plt.figure(0, figsize=(8, 5), frameon=False)
ax = fig.add_subplot(111)

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# horizontal and vertical ticks length
htl, vtl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin), xycoords='data', xy=(0, ymax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

ms = 5
plt.plot(n, xn, 'k.', markersize=ms)
plt.plot(n, sup_xn, 'b.', markersize=ms)
plt.plot(n, inf_xn, 'b.', markersize=ms)

plt.plot([0, nmax], [1, 1], 'r--', lw=2)
plt.plot([0, nmax], [-1, -1], 'r--', lw=2)

plt.annotate(r'$\limsup_n x_n$', xytext=(60, 1.6), xycoords='data', xy=(40, 1),
             textcoords='data', fontsize=fontsize, va="baseline", ha="left", color='red',
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.4", color='red', relpos=(0, 0.5),
                             patchA=None, patchB=None, connectionstyle="arc3,rad=0", shrinkA=5, shrinkB=1))

plt.annotate(r'$\sup_{k\geq n} x_k$', xytext=(120, 1.6), xycoords='data', xy=(105, sup_xn[110]),
             textcoords='data', fontsize=fontsize, va="baseline", ha="left", color='blue',
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.4", color='blue', relpos=(0, 0.5),
                             patchA=None, patchB=None, connectionstyle="arc3,rad=0", shrinkA=5, shrinkB=1))

plt.annotate(r'$x_n$', xytext=(192, 1.6), xycoords='data', xy=(168, xn[168]),
             textcoords='data', fontsize=fontsize, va="baseline", ha="left", color='k',
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.4", color='k', relpos=(0, 0.5),
                             patchA=None, patchB=None, connectionstyle="arc3,rad=0", shrinkA=5, shrinkB=3))

plt.annotate(r'$\liminf_n x_n$', xytext=(75, -1.8), xycoords='data', xy=(55, -1),
             textcoords='data', fontsize=fontsize, va="baseline", ha="left", color='red',
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.4", color='red', relpos=(0, 1),
                             patchA=None, patchB=None, connectionstyle="arc3,rad=0", shrinkA=5, shrinkB=1))

plt.annotate(r'$\inf_{k\geq n} x_k$', xytext=(135, -1.8), xycoords='data', xy=(115, inf_xn[111]),
             textcoords='data', fontsize=fontsize, va="baseline", ha="left", color='blue',
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.4", color='blue', relpos=(0, 1),
                             patchA=None, patchB=None, connectionstyle="arc3,rad=0", shrinkA=5, shrinkB=1))

# ticks and labels
# xlabels
plt.text(xmax, hlm, '$n$', fontsize=12, ha='center', va='baseline')
n_ticks = np.arange(0, nmax, 50)
plt.plot([n_ticks, n_ticks], [0, vtl], 'k')
for n_tick in n_ticks:
    plt.text(n_tick+2, hlm, r'${}$'.format(int(n_tick)), fontsize=10, ha='left', va='baseline')

# ylabels
plt.text(vlm, ymax, r'$x_n$', fontsize=fontsize, ha='left', va='center')
y_ticks = np.arange(-1.5, 2, 0.5)
plt.plot([0, htl], [y_ticks, y_ticks], 'k')
for y_tick in y_ticks:
    plt.text(-vlm, y_tick, r'${0:.2f}$'.format(y_tick), fontsize=10, ha='right', va='center')

plt.axis('off')
# save as pdf image
plt.savefig('liminf_limsup.pdf', bbox_inches='tight')
plt.show()





