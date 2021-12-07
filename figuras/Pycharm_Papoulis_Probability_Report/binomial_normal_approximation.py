import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, binom
import matplotlib.colors as colors
from matplotlib import cm

__author__ = 'ernesto'

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preview'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


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
# PARAMETROS - Puede ser modificado #
#####################################

# parámetros de la va binomial
n = 30
p = 0.6
q = 1 - p

# intervalo
k1 = 15
k2 = 23

#####################
# FIN DE PARAMETROS #
#####################

# media y varianza de la binomial
var = n * p * q
eta = n * p
print(eta)

# axis parameters
nvar = 1.4
xmin = eta - nvar * var
xmax = eta + nvar * var

# valores de la abscisa
x = np.linspace(xmin, xmax, 400)

# densidad normal
fx = norm.pdf(x, loc=eta, scale=np.sqrt(var))

# densidad normal evaluada en los valores de k
ks = np.arange(k1, k2 + 1)
fx_k = norm.pdf(ks, loc=eta, scale=np.sqrt(var))
f_k = binom.pmf(ks, n, p, loc=0)

ymax = np.amax(fx)
ymin = 0

ks_all = np.arange(np.ceil(xmin), np.ceil(xmax))

print("n = {0:d}, p = {1:.1f}".format(n, p))
print("k1 = {0:d}, k2 = {1:d}".format(k1, k2))
print("eta = np = {0:.3f}, var = npq = {1:.3f}".format(eta, var))
print("Valor verdadero: {0:.5f}".format(np.sum(f_k)))
print("Aproximación: {0:.5f}".format(norm.cdf(k2, loc=eta, scale=np.sqrt(var))
                                     - norm.cdf(k1, loc=eta, scale=np.sqrt(var))))
print("Aproximación con corrección: {0:.5f}".format(norm.cdf(k2 + 0.5, loc=eta, scale=np.sqrt(var))
                                                    - norm.cdf(k1 - 0.5, loc=eta, scale=np.sqrt(var))))



#
# PARÁMETROS DE LA FIGURA
#

# maximos y mínimos de los ejes
dx = 1
dy = ymax / 10
xmin_ax = xmin - dx
xmax_ax = xmax + dx
ymin_ax = ymin - dy
ymax_ax = ymax + dy


# margen de etiquetas del eje horizontal
hlm = -0.015
# margen de etiquetas del eje vertical
vlm = -0.02
# tamaño de fuente
fontsize = 13
# largo de los ticks de los ejes (7 pixels)
display_length = 6  # en pixels

# colores
# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col1 = scalarMap.to_rgba(0.8)
col2 = scalarMap.to_rgba(0.2)

##############
#    PLOT    #
##############
fig = plt.figure(0, figsize=(10, 3), frameon=False)

ax = plt.subplot2grid((1, 6), (0, 0), rowspan=1, colspan=3)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
htl, vtl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# density plot
plt.plot(x, fx, 'k', linewidth=2, label=r'$N(np,\,npq)$')
plt.bar(ks[:-1], fx_k[:-1], align='edge', width=1, color=col1, edgecolor='k',
        label=r'$\dfrac{1}{\sqrt{2\pi npq}}e^{-(k-np)^2/2npq}$')
plt.plot(ks, f_k, color=col2, linestyle='None', marker='.', markersize=10,
         label=r'$\displaystyle\binom{n}{k}p^kq^{n-k}$')

# xticks
xticks = np.zeros((2, ks_all.shape[0]))
xticks[1] = vtl
plt.plot([ks_all, ks_all], xticks, 'k-', lw=1)

# ticks labels
# xlabels
plt.text(xmax_ax, hlm, '$x$', fontsize=fontsize, ha='center', va='baseline')
plt.text(k1, hlm, '$k_1$', fontsize=fontsize, ha='center', va='baseline')
plt.text(k2, hlm, '$k_2$', fontsize=fontsize, ha='center', va='baseline')
plt.text(eta, hlm, '$np$', fontsize=fontsize, ha='center', va='baseline')

# legend
leg = plt.legend(loc=(0.75, 0.45), fontsize=fontsize, frameon=False)

plt.axis('off')

# CORRECCIÓN DEL ERROR #
ax = plt.subplot2grid((1, 6), (0, 3), rowspan=1, colspan=3)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# density plot
plt.plot(x, fx, 'k', linewidth=2)
plt.bar(ks, fx_k, align='center', width=1, color=col1, edgecolor='k')
plt.plot(ks, f_k, color=col2, linestyle='None', marker='.', markersize=10)

# xticks
plt.plot([ks_all, ks_all], xticks, 'k-', lw=1)

# ticks labels
# xlabels
plt.text(xmax_ax, hlm, '$x$', fontsize=fontsize, ha='center', va='baseline')
plt.text(k1, hlm, '$k_1$', fontsize=fontsize, ha='center', va='baseline')
plt.text(k2, hlm, '$k_2$', fontsize=fontsize, ha='center', va='baseline')
plt.text(eta, hlm, '$np$', fontsize=fontsize, ha='center', va='baseline')


plt.annotate(r'$k_1-0.5$', xytext=(k1 - 1.4,  2 * hlm), xycoords='data', xy=(k1 - 0.5, 0),
             textcoords='data', fontsize=fontsize, va="baseline", ha="right",
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.4", facecolor='black', relpos=(0.9, 1),
                             patchA=None, patchB=None, shrinkA=4, shrinkB=0))
plt.annotate(r'$k_2+0.5$', xytext=(k2 + 1.2,  2 * hlm), xycoords='data', xy=(k2 + 0.5, 0),
             textcoords='data', fontsize=fontsize, va="baseline", ha="left",
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.4", facecolor='black', relpos=(0.1, 1),
                             patchA=None, patchB=None, shrinkA=4, shrinkB=0))

plt.axis('off')

# save as pdf
plt.savefig('binomial_normal_approximation.pdf', bbox_inches='tight')
plt.show()

