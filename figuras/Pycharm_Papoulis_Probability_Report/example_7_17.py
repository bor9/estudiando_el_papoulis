import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm

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
    yticks_len = data_coords[1, 0] - data_coords[0, 0]
    # transform from display coordinates to data coordinates in y axis
    data_coords = inv.transform([(0, 0), (0, length)])
    # get the length of the segment in data units
    xticks_len = data_coords[1, 1] - data_coords[0, 1]
    return xticks_len, yticks_len


#####################################
# PARAMETROS - Puede ser modificado #
#####################################

# distribución uniforme en (0, T)
T = 1

# range of x of interest
xmin = -1.8 * T
xmax = 1.8 * T

ymin = 0
ymax = 1 / T


#####################
# FIN DE PARAMETROS #
#####################

# parametros de las densidades de x_i: media y varianza
eta_i = 0
var_i = (T ** 2) / 12

# cantidad de variables aleatorias x_i a sumar
n = 3
# media y varianza de la suma
eta = n * eta_i
var = n * var_i

# pdfs
x = np.linspace(xmin, xmax, 400)
# aproximación gaussiana
fnorm = norm.pdf(x, eta, math.sqrt(var))
# pdf verdadera
f = np.zeros(x.shape)
c = 8 * (T ** 3)
idx = np.argwhere((x >= -3*T/2) & (x < -T/2))
f[idx] = np.polyval([4, 12 * T, 9 * (T ** 2)], x[idx]) / c
idx = np.argwhere((x >= -T/2) & (x < T/2))
f[idx] = np.polyval([-8, 0, 6 * (T ** 2)], x[idx]) / c
idx = np.argwhere((x >= T/2) & (x <= 3*T/2))
f[idx] = np.polyval([4, -12 * T, 9 * (T ** 2)], x[idx]) / c
# pdf con corrección de primer orden
m4 = 13 / 80
fbar = fnorm * (1 + (m4 / (var ** 2) - 3) / 24 * np.polyval([1 / (var ** 2), 0, -6 / var, 0, 3], x))

# axis parameters
dx = 0.2
xmin_ax = xmin - dx
xmax_ax = xmax + dx
ymin_ax = ymin - 0.15
ymax_ax = ymax + 0.2

# parámetros de la figura
# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm_pixels = 22
ytm = -0.07
# font size
fontsize = 12
gray = 0.7 * np.ones((3, ))

fig = plt.figure(0, figsize=(6, 4), frameon=False)
ax = plt.subplot2grid((8, 1), (0, 0), rowspan=6, colspan=1)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)
xtm, _ = convert_display_to_data_coordinates(ax.transData, length=xtm_pixels)
xtm = -xtm

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# f_i(x)
plt.plot([-T/2, T/2], [1/T, 1/T], 'b', linewidth=2, label='$f_i(x)$')
plt.plot([T/2, T/2], [0, 1/T], 'b', linewidth=2)
plt.plot([-T/2, -T/2], [0, 1/T], 'b', linewidth=2)
plt.plot([xmin, -T/2], [0, 0], 'b', linewidth=2)
plt.plot([T/2, xmax], [0, 0], 'b', linewidth=2)

# aproximación gaussiana
plt.plot(x, fnorm, 'r', linewidth=1, zorder=5, label='$N\left(0,\,\sigma^2\\right)$')
# pdf verdadera
plt.plot(x, f, color=gray, linewidth=3, zorder=0, label='$f(x)$')
# pdf con corrección
plt.plot(x, fbar, 'k', linewidth=1, zorder=5, label='$\\bar{f}(x)$')


# ticks
plt.plot([T/2, T/2], [0, xtl], 'k')
plt.plot([T, T], [0, xtl], 'k')
plt.plot([1.5*T, 1.5*T], [0, xtl], 'k')
plt.plot([-T/2, -T/2], [0, xtl], 'k')
plt.plot([-T, -T], [0, xtl], 'k')
plt.plot([-1.5*T, -1.5*T], [0, xtl], 'k')
plt.plot([0, ytl], [1/(2*T), 1/(2*T)], 'k')
plt.plot([0, ytl], [1/T, 1/T], 'k')


# labels
# xlables
plt.text(xmax_ax, xtm, '$x$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, xtm, '$0$', fontsize=fontsize, ha='right', va='baseline')
plt.text(T/2, xtm, '${:.1f}$'.format(T/2), fontsize=fontsize, ha='center', va='baseline')
plt.text(T, xtm, '${:.0f}$'.format(T), fontsize=fontsize, ha='center', va='baseline')
plt.text(3*T/2, xtm, '${:.1f}$'.format(3*T/2), fontsize=fontsize, ha='center', va='baseline')
plt.text(-T/2, xtm, '${:.1f}$'.format(-T/2), fontsize=fontsize, ha='center', va='baseline')
plt.text(-T, xtm, '${:.0f}$'.format(-T), fontsize=fontsize, ha='center', va='baseline')
plt.text(-3*T/2, xtm, '${:.1f}$'.format(-3*T/2), fontsize=fontsize, ha='center', va='baseline')

# ylabels
plt.text(ytm, 1/T+0.05, '${:.0f}$'.format(1/T), fontsize=fontsize, ha='right', va='center')
plt.text(ytm, 1/(2*T), '${:.1f}$'.format(1/(2*T)), fontsize=fontsize, ha='right', va='center')

leg = plt.legend(loc='upper right', frameon=False, fontsize=fontsize)

plt.axis('off')

ax = plt.subplot2grid((8, 1), (6, 0), rowspan=2, colspan=1)

ymax_ax = 0.08
ymin_ax = -0.03

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)
xtm, _ = convert_display_to_data_coordinates(ax.transData, length=xtm_pixels)
xtm = -xtm

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# aproximación gaussiana
l1 = plt.plot(x, fnorm-f, 'r', linewidth=2, zorder=5, label='$N\left(0,\,\sigma^2\\right)-f(x)$')
# pdf con corrección
l2 = plt.plot(x, fbar-f, 'k', linewidth=2, zorder=5, label='$\\bar{f}(x)-f(x)$')


# ticks
plt.plot([T/2, T/2], [0, xtl], 'k')
plt.plot([T, T], [0, xtl], 'k')
plt.plot([1.5*T, 1.5*T], [0, xtl], 'k')
plt.plot([-T/2, -T/2], [0, xtl], 'k')
plt.plot([-T, -T], [0, xtl], 'k')
plt.plot([-1.5*T, -1.5*T], [0, xtl], 'k')

plt.plot([0, ytl], [0.05, 0.05], 'k')
plt.plot([0, ytl], [-0.02, -0.02], 'k')


# xlables
plt.text(xmax_ax, xtm, '$x$', fontsize=fontsize, ha='right', va='baseline')

# ylabels
plt.text(ytm, 0.055, '$0.05$', fontsize=fontsize, ha='right', va='center')
plt.text(ytm, -0.025, '$-0.02$', fontsize=fontsize, ha='right', va='center')

# legend
first_legend = plt.legend(l1, ['$N\left(0,\,\sigma^2\\right)-f(x)$'], loc=[0, 0.6], frameon=False, fontsize=fontsize)
plt.gca().add_artist(first_legend)
plt.legend(l2, ['$\\bar{f}(x)-f(x)$'], loc=[0.7, 0.6], frameon=False, fontsize=fontsize)

plt.axis('off')

# save as eps image
plt.savefig('example_7_17.pdf', bbox_inches='tight')
plt.show()


