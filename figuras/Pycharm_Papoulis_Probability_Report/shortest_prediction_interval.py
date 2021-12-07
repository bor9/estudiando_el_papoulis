import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from scipy.optimize import fmin

__author__ = 'ernesto'

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preview'] = True


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
# parámetros alpha y beta
a = 5
b = 2

#####################
# END OF PARAMETERS #
#####################

# axis parameters
xmin = -0.05
xmax = 1.05
ymin = -0.3
ymax = 2.7

xmin_ax = xmin - 0.03
xmax_ax = xmax + 0.06
ymin_ax = ymin
ymax_ax = ymax

# valores de la abscisa
x = np.linspace(xmin, xmax, 400)

# densidad de probabilidad
fx = beta.pdf(x, a, b, loc=0, scale=1)

#
# INTERVALOS QUE NO SE SOLAPAN
#

# coeficiente de confianza
gamma1 = 0.3
# intervalo de confianza mas corto
c1, c2 = shortest_confidence_interval(beta, confidence_coef=gamma1, a=a, b=b)
fc1 = beta.pdf(c1, a, b, loc=0, scale=1)
fc2 = beta.pdf(c2, a, b, loc=0, scale=1)

# moda
xm = (a - 1) / (a + b - 2)
fxm = beta.pdf(xm, a, b, loc=0, scale=1)

# intervalo disjunto con la misma probabilidad
c11 = 0.3
c21 = beta.ppf(beta.cdf(c11, a, b, loc=0, scale=1) + gamma1, a, b, loc=0, scale=1)
fc11 = beta.pdf(c11, a, b, loc=0, scale=1)
fc21 = beta.pdf(c21, a, b, loc=0, scale=1)


# intervalo que produce colas de igual área delta/2
# gamma = 1 - delta
# no se usa. solo para verificar.
d1, d2 = beta.interval(gamma1, a, b, loc=0, scale=1)
fd1 = beta.pdf(d1, a, b, loc=0, scale=1)
fd2 = beta.pdf(d2, a, b, loc=0, scale=1)
# area de las colas - debe ser la misma
# cola inferior
a1 = beta.cdf(d1, a, b, loc=0, scale=1)
# cola superior
a2 = beta.sf(d2, a, b, loc=0, scale=1)

#
# INTERVALOS QUE SE SOLAPAN
#

# coeficiente de confianza
gamma2 = 0.8
# intervalo de confianza mas corto
d1, d2 = shortest_confidence_interval(beta, confidence_coef=gamma2, a=a, b=b)
fd1 = beta.pdf(d1, a, b, loc=0, scale=1)
fd2 = beta.pdf(d2, a, b, loc=0, scale=1)

# intervalo disjunto con la misma probabilidad
d11 = 0.4
d21 = beta.ppf(beta.cdf(d11, a, b, loc=0, scale=1) + gamma2, a, b, loc=0, scale=1)
fd11 = beta.pdf(d11, a, b, loc=0, scale=1)
fd21 = beta.pdf(d21, a, b, loc=0, scale=1)

#
# PARÁMETROS DE LA FIGURA
#

# margen de etiquetas del eje horizontal
hlm = -0.25
# margen de etiquetas del eje vertical
vlm = -0.02
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

# intervalo mas corto
plt.plot([c1, c1], [0, fc1], 'k', linewidth=1)
plt.plot([c2, c2], [0, fc2], 'k', linewidth=1)

# moda
plt.plot([xm, xm], [0, fxm], 'k', linewidth=1)

# intervalo disjunto
plt.plot([c11, c11], [0, fc11], 'k', linewidth=1)
plt.plot([c21, c21], [0, fc21], 'k', linewidth=1)

plt.plot([0, c2], [fc2, fc2], 'k--', linewidth=1)
plt.plot([1, 1], [0, vtl], 'k')

# ticks labels
# xlabels
plt.text(xmax_ax, hlm, '$x$', fontsize=fontsize, ha='right', va='baseline')
plt.text(vlm, hlm, '$0$', fontsize=fontsize, ha='right', va='baseline')
plt.text(c1, hlm, '$c_1$', fontsize=fontsize, ha='center', va='baseline')
plt.text(c2, hlm, '$c_2$', fontsize=fontsize, ha='center', va='baseline')
plt.text(c11, hlm, "$c'_1$", fontsize=fontsize, ha='center', va='baseline')
plt.text(c21, hlm, "$c'_2$", fontsize=fontsize, ha='center', va='baseline')
plt.text(xm, hlm, "$x^*$", fontsize=fontsize, ha='center', va='baseline')
plt.text(1, hlm, "$1$", fontsize=fontsize, ha='center', va='baseline')

# ylabels
plt.text(0.03, ymax_ax, '$f(x)$', fontsize=fontsize, ha='left', va='center')
plt.text(0.05, fc2-0.1, '$f(c_1)=f(c_2)$', fontsize=fontsize, ha='left', va='top')

ax.fill_between(x[np.where((x >= c1) & (x <= c2))], 0, fx[np.where((x >= c1) & (x <= c2))], color=col0, alpha=0.4)
ax.fill_between(x[np.where((x >= c11) & (x <= c21))], 0, fx[np.where((x >= c11) & (x <= c21))], color=col1, alpha=0.4)

# legend
plt.text(xmax_ax, ymax_ax, '$\gamma={:.1f}$'.format(gamma1), fontsize=fontsize, ha='right', va='center')

plt.axis('off')

# INTERVALOS QUE NO SE SOLAPAN #
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
plt.plot([d1, d1], [0, fd1], 'k', linewidth=1)
plt.plot([d2, d2], [0, fd2], 'k', linewidth=1)

# moda
plt.plot([xm, xm], [0, fxm], 'k', linewidth=1)

# intervalo disjunto
plt.plot([d11, d11], [0, fd11], 'k', linewidth=1)
plt.plot([d21, d21], [0, fd21], 'k', linewidth=1)

plt.plot([0, d2], [fd2, fd2], 'k--', linewidth=1)
plt.plot([1, 1], [0, vtl], 'k')

# ticks labels
# xlabels
plt.text(xmax_ax, hlm, '$x$', fontsize=fontsize, ha='right', va='baseline')
plt.text(vlm, hlm, '$0$', fontsize=fontsize, ha='right', va='baseline')
plt.text(d1, hlm, '$c_1$', fontsize=fontsize, ha='center', va='baseline')
plt.text(d2, hlm, '$c_2$', fontsize=fontsize, ha='center', va='baseline')
plt.text(d11, hlm, "$c'_1$", fontsize=fontsize, ha='center', va='baseline')
plt.text(d21, hlm, "$c'_2$", fontsize=fontsize, ha='center', va='baseline')
plt.text(xm, hlm, "$x^*$", fontsize=fontsize, ha='center', va='baseline')
plt.text(1, hlm, "$1$", fontsize=fontsize, ha='center', va='baseline')

# ylabels
plt.text(0.03, ymax_ax, '$f(x)$', fontsize=fontsize, ha='left', va='center')
plt.text(0.05, fd2-0.1, '$f(c_1)=f(c_2)$', fontsize=fontsize, ha='left', va='top')

ax.fill_between(x[np.where((x >= d1) & (x <= d2))], 0, fx[np.where((x >= d1) & (x <= d2))], color=col0, alpha=0.4)
ax.fill_between(x[np.where((x >= d11) & (x <= d21))], 0, fx[np.where((x >= d11) & (x <= d21))], color=col1, alpha=0.4)

# legend
plt.text(xmax_ax, ymax_ax, '$\gamma={:.1f}$'.format(gamma2), fontsize=fontsize, ha='right', va='center')


plt.axis('off')


# save as pdf
plt.savefig('shortest_prediction_interval.pdf', bbox_inches='tight')
plt.show()

