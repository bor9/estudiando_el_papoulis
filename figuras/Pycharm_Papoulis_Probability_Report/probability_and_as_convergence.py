import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import math

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
    yticks_len = data_coords[1, 0] - data_coords[0, 0]
    # transform from display coordinates to data coordinates in y axis
    data_coords = inv.transform([(0, 0), (0, length)])
    # get the length of the segment in data units
    xticks_len = data_coords[1, 1] - data_coords[0, 1]
    return xticks_len, yticks_len


#####################################
# PARAMETERS - This can be modified #
#####################################

N = 40
f0 = 0.15
var_w = 0.06

#####################
# END OF PARAMETERS #
#####################

nf0 = 250
f0s = np.linspace(0, 0.25, nf0)
ns = np.arange(N)


f0 = [0.05, 0.1, 0.14, 0.175, 0.2, 0.37, 0.39, 0.4, 0.45]
offset = 5
np.random.seed(6)
np.random.seed(19)
np.random.seed(23)


nrel = len(f0)
Js = np.zeros((nf0, nrel))
for i in np.arange(nrel):
    w = np.random.normal(loc=0, scale=math.sqrt(var_w), size=(N, ))
    Js[:, i] = np.dot(np.cos(2 * math.pi * np.outer(f0s, ns)), w + np.cos(2 * math.pi * f0[i] * ns)) + offset

Ja = Js[:, :5]
Jb = np.zeros((nf0, 5))
Jb[:, :4] = Js[:, 5:9]

f0aux = [0.05, 0.08, 0.1, 0.14, 0.16, 0.175, 0.195, 0.21]
nrel = len(f0aux)
Js = np.zeros((nf0, nrel))
for i in np.arange(nrel):
    w = np.random.normal(loc=0, scale=math.sqrt(var_w), size=(N, ))
    Js[:, i] = np.dot(np.cos(2 * math.pi * np.outer(f0s, ns)), w + np.cos(2 * math.pi * f0aux[i] * ns))

Jb[:, 4] = 6 * np.mean(Js, axis=1) + offset


# abscissa values
xmin = 0
xmax = 0.25
ymin = 0
ymax = 33
# axis parameters
xmin_ax = xmin
xmax_ax = xmax + 0.02
ymax_ax = ymax
ymin_ax = ymin


# x ticks labels margin
xtm = -3
ytm = 0.007
# font size
fontsize = 14

epsilon = 10

# GRAFICAS
fig = plt.figure(0, figsize=(10, 3), frameon=False)
# SUBPLOT 1
ax = plt.subplot2grid((1, 4), (0, 0), rowspan=1, colspan=2)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)


# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(f0s, Ja)

plt.text(ytm, ymax_ax, '$|\mathbf{x}_n-\mathbf{x}|$', fontsize=fontsize, ha='left', va='center')
plt.text(xmax_ax, xtm, '$n$', fontsize=fontsize, ha='center', va='baseline')
plt.text(xmin, xtm, '$n_0$', fontsize=fontsize, ha='center', va='baseline')

plt.plot([xmin, xmax], [epsilon, epsilon], 'k--')
plt.text(xmin-ytm, epsilon, '$\epsilon$', fontsize=fontsize, ha='right', va='center')

plt.axis('off')


ax = plt.subplot2grid((1, 4), (0, 2), rowspan=1, colspan=2)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))


plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(f0s, Jb)

plt.text(ytm, ymax_ax, '$|\mathbf{x}_n-\mathbf{x}|$', fontsize=fontsize, ha='left', va='center')
plt.text(xmax_ax, xtm, '$n$', fontsize=fontsize, ha='center', va='baseline')
plt.text(xmin, xtm, '$n_0$', fontsize=fontsize, ha='center', va='baseline')

plt.plot([xmin, xmax], [epsilon, epsilon], 'k--')
plt.text(xmin-ytm, epsilon, '$\epsilon$', fontsize=fontsize, ha='right', va='center')

plt.axis('off')

leg = plt.legend(['$\zeta_1$', '$\zeta_2$', '$\zeta_3$', '$\zeta_4$', '$\zeta_5$'], loc=(0.35, 0.8),
                 frameon=False, fontsize=fontsize, ncol=3, handlelength=1)


# save as pdf image
plt.savefig('probability_and_as_convergence.pdf', bbox_inches='tight')

plt.show()
