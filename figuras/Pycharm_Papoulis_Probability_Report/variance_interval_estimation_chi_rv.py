import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from scipy.optimize import fmin

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


def shortest_confidence_interval(dist_name, confidence_coef=0.95, **args):
    """ Cálculo del intervalo de confianza mas corto.

    :param dist_name: nombre scipy de la distribución (str)
    :param confidence_coef: coeficiente de confianza (float entre 0 y 1)
    :param args: argumentos scipy de la distribución
    :return: límite inferior y superior del intervalo (array [low, high])

    Ejemplo de uso:

    c1, c2 = shortest_confidence_interval(beta, confidence_coef=0.9, a=5, b=3)
    """
    # configuración de la distribución con los argumentos dados
    distri = dist_name(**args)
    # estimación inicial de la probabilidad (área) de la cola inferior
    low_tail_pr_ini = 1.0 - confidence_coef

    def interval_width(low_tail_pr):
        """
        Cálculo del intervalo de confianza cuando el área de la cola inferior es low_tail_pr.

        :param low_tail_pr: probabilidad de la cola inferior (float entre 0 y 1)
        :return: largo del intervalo (float)
        """
        return distri.ppf(confidence_coef + low_tail_pr) - distri.ppf(low_tail_pr)

    # buscar la probabilidad de la cola inferior (low_tail_pr) que minimiza el intervalo de
    # confianza (función interval_width) fijando el coeficiente de confianza (confidence_coef).
    low_tail_pr_shortest_interval = fmin(interval_width, low_tail_pr_ini, ftol=1e-8, disp=False)[0]
    # se retorna el intervalo como un array([low, high])
    return distri.ppf([low_tail_pr_shortest_interval, confidence_coef + low_tail_pr_shortest_interval])


#####################################
# PARAMETERS - This can be modified #
#####################################

# densidad de probabilidad beta
# parámetro n de la densidad chi
n = 6
# coeficiente de confianza
gamma = 0.8

#####################
# END OF PARAMETERS #
#####################

# nivel de confianza
delta = 1 - gamma

# valores del eje x
xmin = 0
xmax = 20

# valores de la abscisa
x = np.linspace(xmin, xmax, 400)
# densidad de probabilidad
fx = chi2.pdf(x, n)

ymin = 0
ymax = np.amax(fx)

# limites de los ejes de la gráfica
dx = xmax / 14
dy = ymax / 10
xmin_ax = xmin - dx
xmax_ax = xmax + dx
ymin_ax = ymin - dy
ymax_ax = ymax + dy

# intervalo que produce colas de igual área delta/2
d1, d2 = chi2.interval(gamma, n)
fd1 = chi2.pdf(d1, n)
fd2 = chi2.pdf(d2, n)
# area de las colas - debe ser la misma
# cola inferior
a1 = chi2.cdf(d1, n)
# cola superior
a2 = chi2.sf(d2, n)


# intervalo de confianza mas corto
c1, c2 = shortest_confidence_interval(chi2, confidence_coef=gamma, df=n)
fc1 = chi2.pdf(c1, n)
fc2 = chi2.pdf(c2, n)

print("Parámetros: n = {0:d}, gamma = {1:.2f}".format(n, gamma))
print("Largo del intervalo de colas de igual área: {0:.2f}".format(d2 - d1))
print("Intervalo mas corto: {0:.2f}".format(c2 - c1))

#
# PARÁMETROS DE LA FIGURA
#

# margen de etiquetas del eje horizontal
hlm = -0.015
# margen de etiquetas del eje vertical
vlm = -0.5
# tamaño de fuente
fontsize = 14
# largo de los ticks de los ejes (7 pixels)
display_length = 7  # en pixels

# colores
col0 = '#ff000d'
col1 = '#0343df'

##############
#    PLOT    #
##############
fig = plt.figure(0, figsize=(10, 3), frameon=False)

# INTERVALOS QUE NO SE SOLAPAN #
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
plt.plot(x, fx, 'k', linewidth=2)

# intervalo de colas de igual area
plt.plot([d1, d1], [0, fd1], 'k', linewidth=2)
plt.plot([d2, d2], [0, fd2], 'k', linewidth=2)

# tick en la media
plt.plot([n, n], [0, vtl], 'k-', lw=1)


# ticks labels
# xlabels
plt.text(xmax_ax, hlm, '$x$', fontsize=fontsize, ha='right', va='baseline')
plt.text(vlm, hlm, '$0$', fontsize=fontsize, ha='right', va='baseline')
plt.text(d1 + 0.5, hlm, '$\chi^2_{\delta/2}(n)$', fontsize=fontsize, ha='center', va='baseline')
plt.text(d2, hlm, '$\chi^2_{1-\delta/2}(n)$', fontsize=fontsize, ha='center', va='baseline')
plt.text(n, hlm, '$n$', fontsize=fontsize, ha='center', va='baseline')


# ylabels
plt.text(-vlm, ymax_ax, '$\chi^2(n)$', fontsize=fontsize, ha='left', va='center')

plt.text((d1 + d2) / 2, 0.025, '$\gamma=1-\delta$', fontsize=fontsize, ha='center', va='baseline')

ax.fill_between(x[np.where((x >= 0) & (x <= d1))], 0, fx[np.where((x >= 0) & (x <= d1))], color=col0, alpha=0.4)
ax.fill_between(x[np.where((x >= d2) & (x <= xmax))], 0, fx[np.where((x >= d2) & (x <= xmax))], color=col0, alpha=0.4)

plt.annotate(r'$\dfrac{\delta}{2}$', xytext=(-0.75,  0.075), xycoords='data', xy=(3 * d1 / 4, 0.01),
             textcoords='data', fontsize=fontsize, va="baseline", ha="right",
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.4", facecolor='black', relpos=(0.9, 0),
                             patchA=None, patchB=None, shrinkA=4, shrinkB=0))
xx1 = d2 + 1
xx2 = xx1 + 1
plt.annotate(r'$\dfrac{\delta}{2}$', xytext=(xx2 + 0.75,  0.075), xycoords='data', xy=(xx1, 0.01),
             textcoords='data', fontsize=fontsize, va="baseline", ha="left",
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.4", facecolor='black', relpos=(0, 0),
                             patchA=None, patchB=None, shrinkA=4, shrinkB=0))

plt.axis('off')

#
ax = plt.subplot2grid((1, 6), (0, 3), rowspan=1, colspan=3)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# density plot
plt.plot(x, fx, 'k', linewidth=2)

# intervalo mas corto
plt.plot([c1, c1], [0, fc1], 'k', linewidth=2)
plt.plot([c2, c2], [0, fc2], 'k', linewidth=2)

plt.plot([0, c2], [fc2, fc2], 'k--', linewidth=1.5)

# tick en la media
plt.plot([n, n], [0, vtl], 'k-', lw=1)


# ticks labels
# xlabels
plt.text(xmax_ax, hlm, '$x$', fontsize=fontsize, ha='right', va='baseline')
plt.text(vlm, hlm, '$0$', fontsize=fontsize, ha='right', va='baseline')
plt.text(c1, hlm, '$c_1$', fontsize=fontsize, ha='center', va='baseline')
plt.text(c2, hlm, '$c_2$', fontsize=fontsize, ha='center', va='baseline')
plt.text(n, hlm, '$n$', fontsize=fontsize, ha='center', va='baseline')

# ylabels
plt.text(-vlm, ymax_ax, '$\chi^2(n)$', fontsize=fontsize, ha='left', va='center')

plt.text((c1 + c2) / 2, 0.025, '$\gamma=1-\delta$', fontsize=fontsize, ha='center', va='baseline')

plt.axis('off')

# save as pdf
plt.savefig('variance_interval_estimation_chi_rv.pdf', bbox_inches='tight')
plt.show()

