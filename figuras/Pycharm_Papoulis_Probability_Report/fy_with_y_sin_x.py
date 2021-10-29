import matplotlib.pyplot as plt
import numpy as np
import math

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


def curly_brace(xmin, xmax, y, amp, position='up', beta=8, step=0.01):
    xm = (xmin+xmax)/2
    x_left = np.arange(xmin, xm, step)
    hb_left = amp*(1/(1.+np.exp(-1*beta*(x_left-xmin)))+1/(1.+np.exp(-1*beta*(x_left-xm)))-1/2)
    x = np.concatenate((x_left, np.arange(xm, xmax-step, step)))
    hb = np.concatenate((hb_left, hb_left[-2::-1]))
    if position == 'up':
        return x, hb+y
    elif position == 'down':
        return x, -hb+y
    elif position == 'left':
        return hb, x
    elif position == 'right':
        return -hb, x

#####################################
# PARAMETERS - This can be modified #
#####################################

# sinusoid period and phase (samples)
N = 200
n0 = 14
# sinusoid amplitude
a = 1
# y value
y0 = 0.75

# first and last sample number
nmin = -190
nmax = 270

# y axis maximum value
ymax = 1.4
ymin = -ymax

# extension to the left and right of the range of interest
delta_n = 15


#####################
# END OF PARAMETERS #
#####################

# samples
n = np.arange(nmin, nmax)

# sinusoid angular frequency (rad)
theta = 2 * math.pi / N
# sinusoid phase (rad)
phi = 2 * math.pi * n0 / N
# sinusoid
x = theta * n
x_fir = x[0]
x_las = x[-1]
y = a * np.sin(x + phi)

k = np.arange(-2, 4)
nk = k.size
x_k = np.zeros(k.shape)
for i in np.arange(nk):
    if k[i] % 2 == 0:
        x_k[i] = math.asin(y0 / a) - phi + k[i] * math.pi
    else:
        x_k[i] = -math.asin(y0 / a) - phi + k[i] * math.pi


# axis parameters
delta_x = delta_n * theta
xmin = x_fir - delta_x
xmax = x_las + delta_x


# horizontal label margin
hlm = -0.2
# horizontal label margin
vlm = -0.1
# font size
fontsize = 14
# dashes length/space
dashed = (4, 4)
# length of the ticks for all subplot (7 pixels)
display_length = 6  # in pixels

fig = plt.figure(0, figsize=(8, 3), frameon=False)
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

plt.plot([x_fir, x_las], [y0, y0], 'r', linewidth=1)
plt.plot(x, y, 'k', linewidth=2)

# solutions marks and labels
for xki in x_k:
    plt.plot([xki, xki], [0, y0], 'k--', linewidth=0.8, dashes=dashed)
plt.plot(x_k, y0*np.ones(x_k.shape), 'k.', markersize=10)
for i in [0, 1, 4, 5]:
    plt.text(x_k[i], hlm, r'$x_{{{}}}$'.format(k[i]), fontsize=fontsize, ha='center', va='baseline')

# xticks
plt.plot([math.pi, math.pi], [0, vtl], 'k')
plt.plot([-math.pi, -math.pi], [0, vtl], 'k')
plt.plot([-phi, -phi], [0, vtl], 'k')

# xlabels and xtickslabels
plt.text(xmax, hlm, '$x$', fontsize=fontsize, ha='right', va='baseline')
plt.text(x_k[2], hlm, '$x_0$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-phi*1.5, -hlm*0.7, r'$-\theta$', fontsize=fontsize, ha='center', va='baseline')
plt.text(math.pi, hlm, '$\pi$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-math.pi, -hlm*0.7, '$-\pi$', fontsize=fontsize, ha='center', va='baseline')

plt.annotate(r'$\pi-\theta$', xytext=(math.pi, 0.5), xycoords='data', xy=(math.pi-phi, 0), textcoords='data',
             fontsize=fontsize,
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.4", facecolor='black', relpos=(0.1, 0),
                             shrinkA=0, shrinkB=1))


x2, b2 = curly_brace(x_k[3], math.pi-phi, hlm/2, 0.15, position='down', beta=60, step=0.01)
plt.plot(x2, b2, 'k')
plt.text((x_k[3]+math.pi-phi)/2, -0.28, r'$x_0+\theta$', fontsize=fontsize, ha='center', va='top')

plt.annotate(r'$x_1=\pi-x_0-2\theta$', xytext=(0.15, -0.8), xycoords='data', xy=(math.pi-x_k[2]-2*phi, 0),
             textcoords='data', fontsize=fontsize, va="top", ha="left",
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.4", facecolor='black', relpos=(0.12, 1),
                             patchA=None, patchB=None, connectionstyle="arc3,rad=0", shrinkB=1))


# yticks and ytickslabels
plt.plot([0, htl], [1, 1], 'k')
plt.text(vlm, 1, '$a$', fontsize=fontsize, ha='right', va='center')
plt.annotate(r'$y$', xytext=(-1, 1), xycoords='data', xy=(0, y0), textcoords='data',
             fontsize=fontsize,  va="baseline", ha="right",
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.4", facecolor='black', relpos=(0, 0.5),
                             shrinkA=0, shrinkB=1))

# ylabel
plt.text(-2*vlm, ymax, r'$g(x)=a\,\sin(x+\theta)$', fontsize=fontsize, ha='left', va='center')

plt.axis('off')
# save as eps image
plt.savefig('fy_with_y_sin_x.pdf', bbox_inches='tight')
plt.show()





