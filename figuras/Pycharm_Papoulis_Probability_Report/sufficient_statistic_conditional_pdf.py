import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
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

# media y varianza de pdf normal
var = 0.5
eta = 3

#####################
# END OF PARAMETERS #
#####################

# abscissa values
xmin = -0.1
xmax = 2 * eta

x = np.linspace(xmin, xmax, 300)
# verosimilitud - no suficiente
pdf_ns = norm.pdf(x, eta, math.sqrt(var))

# axis parameters
dx = xmax / 15
xmin_ax = xmin - dx
xmax_ax = xmax + dx

ym = np.amax(pdf_ns)
ymax_ax = ym + ym / 4
ymin_ax = -ym / 8


# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.08
ytm = 0.2
# font size
fontsize = 14


fig = plt.figure(0, figsize=(10, 3), frameon=False)

# PLOT OF F(x | x < a)
ax = plt.subplot2grid((1, 8), (0, 0), rowspan=1, colspan=4)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
htl, vtl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(x, pdf_ns, color='k', linewidth=2)

# xlabels and xtickslabels
plt.plot([eta, eta], [0, vtl], 'k')
plt.text(eta, xtm, r'$\theta_0$', fontsize=fontsize, ha='center', va='baseline')
plt.text(xmax_ax, xtm, r'$\theta$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, ymax_ax, r'$P\{{\bf x}_1=x_1,\,\dots,\,{\bf x}_n=x_n\,;\theta\,|\,T_1({\bf x})=t\}$',
         fontsize=fontsize, ha='left', va='center')

plt.axis('off')

##
ax = plt.subplot2grid((1, 8), (0, 4), rowspan=1, colspan=4)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

pdf_s = 2 * ym / 3
plt.plot([xmin, xmax], [pdf_s, pdf_s], color='k', linewidth=2)

# xlabels and xtickslabels
plt.text(xmax_ax, xtm, r'$\theta$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, ymax_ax, r'$P\{{\bf x}_1=x_1,\,\dots,\,{\bf x}_n=x_n\,;\theta\,|\,T_2({\bf x})=t\}$',
         fontsize=fontsize, ha='left', va='center')

plt.axis('off')

# save as pdf image
plt.savefig('sufficient_statistic_conditional_pdf.pdf', bbox_inches='tight')

plt.show()

