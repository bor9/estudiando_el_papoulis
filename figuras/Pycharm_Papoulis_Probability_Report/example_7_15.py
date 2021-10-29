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
T = 0.5

# range of x of interest
xmin = -0.1
xmax = 3.5 * T

ymin = 0
ymax = 1 / T


#####################
# FIN DE PARAMETROS #
#####################

# parametros de las densidades de x_i: media y varianza
eta = T / 2
var = (T ** 2) / 12

# cantidad de variables aleatorias x_i a sumar
na = 2
nb = 3
# media y varianza de la suma
eta2 = na * eta
var2 = na * var
eta3 = nb * eta
var3 = nb * var

# pdf teorica
x = np.linspace(xmin, xmax, 300)
f2 = norm.pdf(x, eta2, math.sqrt(var2))
f3 = norm.pdf(x, eta3, math.sqrt(var3))


# axis parameters
dx = 0.1
xmin_ax = xmin - dx
xmax_ax = xmax + 2 * dx
dy = 0.2
ymin_ax = ymin - dy
ymax_ax = ymax + 0.4

# parámetros de la figura
# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.23
ytm = -0.07
# font size
fontsize = 14

fig = plt.figure(0, figsize=(10, 3), frameon=False)
ax = plt.subplot2grid((1, 6), (0, 0), rowspan=1, colspan=2)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# f(x)
plt.plot([0, T], [1/T, 1/T], 'k', linewidth=2)
plt.plot([T, T], [0, 1/T], 'k', linewidth=2)
plt.plot([0, 0], [0, 1/T], 'k', linewidth=2)
plt.plot([xmin, 0], [0, 0], 'k', linewidth=2)
plt.plot([T, xmax], [0, 0], 'k', linewidth=2)

# labels
# xlables
plt.text(xmax_ax, xtm, '$x$', fontsize=fontsize, ha='right', va='baseline')
plt.text(T, xtm, '$T$', fontsize=fontsize, ha='center', va='baseline')
plt.text(ytm, xtm, '$0$', fontsize=fontsize, ha='right', va='baseline')
# ylabels
plt.text(ytm, 1/T, '$\dfrac{1}{T}$', fontsize=fontsize, ha='right', va='center')
plt.text(-ytm, ymax_ax, '$f(x)$', fontsize=fontsize, ha='left', va='center')

plt.axis('off')


fig = plt.figure(0, figsize=(10, 3), frameon=False)
ax = plt.subplot2grid((1, 6), (0, 2), rowspan=1, colspan=2)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# f2(x)
plt.plot([0, T], [0, 1/T], 'k', linewidth=2, label='$f(x)*f(x)$')
plt.plot([T, 2 * T], [1/T, 0], 'k', linewidth=2)
plt.plot([xmin, 0], [0, 0], 'k', linewidth=2)
plt.plot([2*T, xmax], [0, 0], 'k', linewidth=2)
# aproximación gaussiana
plt.plot(x, f2, 'r', linewidth=2, zorder=0, label='$N\left(T,\,\dfrac{T^2}{6}\\right)$')

# ticks
plt.plot([T, T], [0, xtl], 'k')
plt.plot([2*T, 2*T], [0, xtl], 'k')
plt.plot([0, ytl], [1/T, 1/T], 'k')

# labels
# xlables
plt.text(xmax_ax, xtm, '$x$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, xtm, '$0$', fontsize=fontsize, ha='right', va='baseline')
plt.text(T, xtm, '$T$', fontsize=fontsize, ha='center', va='baseline')
plt.text(2*T, xtm, '$2T$', fontsize=fontsize, ha='center', va='baseline')
# ylabels
plt.text(ytm, 1/T, '$\dfrac{1}{T}$', fontsize=fontsize, ha='right', va='center')
#plt.text(-ytm, ymax_ax, '$f_2(x)$', fontsize=fontsize, ha='left', va='center')

leg = leg = plt.legend(loc=(0.45, 0.7), frameon=False, fontsize=12)

plt.axis('off')


fig = plt.figure(0, figsize=(10, 3), frameon=False)
ax = plt.subplot2grid((1, 6), (0, 4), rowspan=1, colspan=2)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# f3(x)
c = 2 * (T ** 3)
xa = np.linspace(0, T, 100)
plt.plot(xa, np.polyval([1, 0, 0], xa) / c, 'k', linewidth=2, label='$f(x)*f(x)*f(x)$')
xa = np.linspace(T, 2 * T, 100)
plt.plot(xa, np.polyval([-2, 6 * T, -3 * (T ** 2)], xa) / c, 'k', linewidth=2)
xa = np.linspace(2 * T, 3 * T, 100)
plt.plot(xa, np.polyval([1, -6 * T, 9 * (T ** 2)], xa) / c, 'k', linewidth=2)

plt.plot([xmin, 0], [0, 0], 'k', linewidth=2)
plt.plot([3*T, xmax], [0, 0], 'k', linewidth=2)
# aproximación gaussiana
plt.plot(x, f3, 'r', linewidth=2, zorder=0, label='$N\left(\dfrac{3T}{2},\,\dfrac{T^2}{4}\\right)$')

# ticks
plt.plot([T, T], [0, xtl], 'k')
plt.plot([2*T, 2*T], [0, xtl], 'k')
plt.plot([3*T, 3*T], [0, xtl], 'k')
plt.plot([0, ytl], [1/T, 1/T], 'k')
plt.plot([0, ytl], [1/(2*T), 1/(2*T)], 'k')

# labels
# xlables
plt.text(xmax_ax, xtm, '$x$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, xtm, '$0$', fontsize=fontsize, ha='right', va='baseline')
plt.text(T, xtm, '$T$', fontsize=fontsize, ha='center', va='baseline')
plt.text(2*T, xtm, '$2T$', fontsize=fontsize, ha='center', va='baseline')
plt.text(3*T, xtm, '$3T$', fontsize=fontsize, ha='center', va='baseline')
# ylabels
plt.text(ytm, 1/T, '$\dfrac{1}{T}$', fontsize=fontsize, ha='right', va='center')
plt.text(ytm, 1/(2*T), '$\dfrac{1}{2T}$', fontsize=fontsize, ha='right', va='center')
#plt.text(-ytm, ymax_ax, '$f_3(x)$', fontsize=fontsize, ha='left', va='center')

leg = leg = plt.legend(loc=(0.28, 0.7), frameon=False, fontsize=12)

plt.axis('off')

# save as eps image
plt.savefig('example_7_15.pdf', bbox_inches='tight')
plt.show()


