import matplotlib.pyplot as plt
import numpy as np

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
# PARAMETERS - This can be modified #
#####################################

# varianza del ruido
q = 2
# parámetro de la respuesta al impulso
c = 1.2

t1 = 0.5
t2 = 2

#####################
# END OF PARAMETERS #
#####################

# abscissa values
xmin = -0.5
xmax = 4

x2 = np.linspace(t1, xmax, 300)
rxy2 = q * np.exp(-c * (x2 - t1))

x1 = np.linspace(0, t2, 300)
rxy1 = q * np.exp(-c * (t2 - x1))

# axis parameters
dx = 0.3
xmin_ax = xmin - dx
xmax_ax = xmax + dx

ym = q + 0.2
ymax_ax = ym + ym / 4
ymin_ax = -ym / 6


# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.37
ytm = 0.13
# font size
fontsize = 14


fig = plt.figure(0, figsize=(10, 2.5), frameon=False)

# Gráfica de R_xy(t1, t2) como función de t2
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

plt.plot(x2, rxy2, color='k', linewidth=2)
plt.plot([xmin, t1], [0, 0], color='k', linewidth=2)

plt.plot([t1, t1], [0, q], 'k--', dashes=(2, 1))

# labels
plt.plot([0, htl], [q, q], 'k')
plt.text(-ytm, q, r'$q$', fontsize=fontsize, ha='right', va='center')
plt.text(t1, xtm, r'$t_1$', fontsize=fontsize, ha='center', va='baseline')
plt.text(xmax_ax, xtm, r'$t_2$', fontsize=fontsize, ha='right', va='baseline')
plt.text(-ytm, xtm, r'$0$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, ymax_ax, r'$R_{xy}(t_1,\,t_2)\textrm{ como funci\'on de }t_2$', fontsize=fontsize, ha='left', va='center')

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

plt.plot(x1, rxy1, color='k', linewidth=2)
plt.plot([xmin, 0], [0, 0], color='k', linewidth=2)
plt.plot([t2, xmax], [0, 0], color='k', linewidth=2)

plt.plot([t2, t2], [0, q], 'k--', dashes=(2, 1))

# labels
plt.plot([0, htl], [q, q], 'k')
plt.text(-ytm, q, r'$q$', fontsize=fontsize, ha='right', va='center')
plt.text(t2, xtm, r'$t_2$', fontsize=fontsize, ha='center', va='baseline')
plt.text(xmax_ax, xtm, r'$t_1$', fontsize=fontsize, ha='right', va='baseline')
plt.text(-ytm, xtm, r'$0$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, ymax_ax, r'$R_{xy}(t_1,\,t_2)\textrm{ como funci\'on de }t_1$', fontsize=fontsize, ha='left', va='center')

plt.axis('off')

# save as pdf image
plt.savefig('example_9_18_1.pdf', bbox_inches='tight')


fig = plt.figure(1, figsize=(10, 5), frameon=False)

# Gráfica de R_xy(-alpha, t2) como función de alpha
xmin1 = -3
xmax1 = 3
dx = 0.4
xmin_ax1 = xmin1 - dx
xmax_ax1 = xmax1 + dx

t1 = 1.2
xa = np.linspace(-t2, 0, 300)
rxya = q * np.exp(-c * (t2 + xa))
xb = np.linspace(t1-t2, t1, 300)
rxyb = q * np.exp(-c * (t2 - (t1 - xb)))


ax = plt.subplot2grid((2, 8), (0, 2), rowspan=1, colspan=4)
plt.xlim(xmin_ax1, xmax_ax1)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax1, 0), xycoords='data', xy=(xmax_ax1, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(xa, rxya, color='k', linewidth=2)
plt.plot([xmin1, -t2], [0, 0], color='k', linewidth=2)
plt.plot([0, xmax1], [0, 0], color='k', linewidth=2)

plt.plot([-t2, -t2], [0, q], 'k--', dashes=(2, 1))

# labels
plt.plot([0, htl], [q, q], 'k')
plt.text(-ytm, q, r'$q$', fontsize=fontsize, ha='right', va='center')
plt.text(-t2, xtm, r'$-t_2$', fontsize=fontsize, ha='center', va='baseline')
plt.text(xmax_ax1, xtm, r'$\alpha$', fontsize=fontsize, ha='right', va='baseline')
plt.text(-ytm, xtm, r'$0$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, ymax_ax, r'$R_{xy}(-\alpha,\,t_2)$', fontsize=fontsize, ha='left', va='center')
plt.axis('off')


ax = plt.subplot2grid((2, 8), (1, 0), rowspan=1, colspan=4)
plt.xlim(xmin_ax1, xmax_ax1)
plt.ylim(ymin_ax, ymax_ax)

# axis arrows
plt.annotate("", xytext=(xmin_ax1, 0), xycoords='data', xy=(xmax_ax1, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(xb, rxyb, color='k', linewidth=2)
plt.plot([xmin1, t1-t2], [0, 0], color='k', linewidth=2)
plt.plot([t1, xmax1], [0, 0], color='k', linewidth=2)

plt.plot([t1-t2, t1-t2], [0, q], 'k--', dashes=(2, 1))
plt.plot([t1, t1], [0, rxyb[-1]], 'k--', dashes=(2, 1))

# labels
plt.plot([0, htl], [q, q], 'k')
plt.text(-ytm, q, r'$q$', fontsize=fontsize, ha='right', va='center')
plt.text(t1-t2, xtm, r'$t_1-t_2$', fontsize=fontsize, ha='center', va='baseline')
plt.text(t1, xtm, r'$t_1$', fontsize=fontsize, ha='center', va='baseline')
plt.text(xmax_ax1, xtm, r'$\alpha$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, ymax_ax, r'$R_{xy}(t_1-\alpha,\,t_2)$', fontsize=fontsize, ha='left', va='center')
plt.text(xmin_ax1, ymax_ax, r'$t_1<t_2$', fontsize=fontsize, ha='left', va='center')

plt.axis('off')

t1 = 2.7
xb = np.linspace(t1-t2, t1, 300)
rxyb = q * np.exp(-c * (t2 - (t1 - xb)))

ax = plt.subplot2grid((2, 8), (1, 4), rowspan=1, colspan=4)
plt.xlim(xmin_ax1, xmax_ax1)
plt.ylim(ymin_ax, ymax_ax)

# axis arrows
plt.annotate("", xytext=(xmin_ax1, 0), xycoords='data', xy=(xmax_ax1, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(xb, rxyb, color='k', linewidth=2)
plt.plot([xmin1, t1-t2], [0, 0], color='k', linewidth=2)
plt.plot([t1, xmax1], [0, 0], color='k', linewidth=2)

plt.plot([t1-t2, t1-t2], [0, q], 'k--', dashes=(2, 1))
plt.plot([t1, t1], [0, rxyb[-1]], 'k--', dashes=(2, 1))

# labels
plt.plot([0, htl], [q, q], 'k')
plt.text(-ytm, q, r'$q$', fontsize=fontsize, ha='right', va='center')
plt.text(t1-t2, xtm, r'$t_1-t_2$', fontsize=fontsize, ha='center', va='baseline')
plt.text(t1, xtm, r'$t_1$', fontsize=fontsize, ha='center', va='baseline')
plt.text(xmax_ax1, xtm, r'$\alpha$', fontsize=fontsize, ha='right', va='baseline')
plt.text(-ytm, xtm, r'$0$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, ymax_ax, r'$R_{xy}(t_1-\alpha,\,t_2)$', fontsize=fontsize, ha='left', va='center')
plt.text(xmin_ax1, ymax_ax, r'$t_1>t_2$', fontsize=fontsize, ha='left', va='center')

plt.axis('off')

# save as pdf image
plt.savefig('example_9_18_2.pdf', bbox_inches='tight')

plt.show()