import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)


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

# f(x): normal density
mean = 1
variance = 1

# number of samples for the functions plot
N = 200
# number of trials
n = 12

# range of x axis
xmin = -2
xmax = 4
# range of y axis
ymax = 1.2

# normal distribution samples
x = np.linspace(xmin, xmax, N)
fx = norm.pdf(x, loc=mean, scale=variance)  # density
Fx = norm.cdf(x, loc=mean, scale=variance)  # distribution
q = np.linspace(Fx[0], Fx[-1], N)
invFx = norm.ppf(q, loc=mean, scale=variance)  # percentil

# trials
# rs = 7, 38, 40, 62
xi = norm.rvs(loc=mean, scale=variance, size=n, random_state=62)
xi = np.sort(xi)
xi[0] = -1
xi[4] = 0.9
inv_n = np.arange(1, n+1)/n

# extension to the left and right of the range of interest
delta_x = 0.4
delta_y = 0
x_ax_min = xmin - delta_x
x_ax_max = xmax + delta_x
y_ax_max = ymax + delta_y
y_ax_min = x_ax_min * y_ax_max / x_ax_max  # for x axis in the same line in the two plots
ymin = y_ax_min + delta_y
# length of the ticks for all subplot in pixels
ticks_length = 5

x_bl1 = -0.12
x_bl2 = x_bl1 * (x_ax_max-x_ax_min) / (y_ax_max-y_ax_min)
y_rm1 = -0.1
y_rm2 = x_bl1 * (y_ax_max-y_ax_min) / (x_ax_max-x_ax_min)
font_size = 14
grey = '#888888'

fig = plt.figure(0, figsize=(10, 4), frameon=False)
# SUBPLOT 1
ax = plt.subplot2grid((1, 10), (0, 0), rowspan=1, colspan=5)
plt.axis([x_ax_min, x_ax_max, y_ax_min, y_ax_max])
ax.set_aspect(abs(x_ax_max-x_ax_min)/abs(y_ax_max-y_ax_min))

# horizontal and vertical ticks length
htl, vtl = convert_display_to_data_coordinates(ax.transData, length=ticks_length)

# axis arrows
plt.annotate("", xytext=(x_ax_min, 0), xycoords='data', xy=(x_ax_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=5, headlength=7, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, y_ax_min), xycoords='data', xy=(0, y_ax_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=5, headlength=7, facecolor='black', shrink=0.002))

# axis labels
plt.text(x_ax_max, x_bl1, r'$x$', fontsize=font_size, ha='right', va='baseline')

plt.plot(x, Fx, color=grey, lw=2)
plt.plot(xi, inv_n, 'k.', markersize=6)

xi_ext = np.concatenate([[xmin], xi, [xmax]])
inv_n_ext = np.concatenate([[0], inv_n])

plt.plot([xi_ext[:-1], xi_ext[1:]], [inv_n_ext, inv_n_ext], 'k')
plt.plot([xi, xi], [inv_n_ext[:-1], inv_n_ext[1:]], 'k')

# ticks
plt.plot([xi, xi], [0, vtl], 'k')
plt.plot([0, htl], [inv_n, inv_n], 'k')

plt.text(xi[0], x_bl1, r'$x_{\mathrm{min}}$', fontsize=font_size, ha='center', va='baseline')
plt.text(xi[-1]+0.2, x_bl1, r'$x_{\mathrm{max}}$', fontsize=font_size, ha='center', va='baseline')
i1 = 8
plt.text(xi[i1]+0.1, x_bl1, r'$x_i$', fontsize=font_size, ha='center', va='baseline')

plt.text(y_rm1, 1, r'$1$', fontsize=font_size, ha='right', va='center')
plt.text(y_rm1, x_bl1, r'$0$', fontsize=font_size, ha='right', va='baseline')

# 1/n
yd = -0.3
i2 = 6
plt.plot([yd-2*vtl, yd+2*vtl], [inv_n[i2], inv_n[i2]], 'k')
plt.plot([yd-2*vtl, yd+2*vtl], [inv_n[i2+1], inv_n[i2+1]], 'k')
plt.plot([yd, yd], [inv_n[i2], inv_n[i2+1]], 'k')
plt.text(yd+y_rm1, (inv_n[i2]+inv_n[i2+1])/2, r'$\frac{1}{n}$', fontsize=font_size, ha='right', va='center')

plt.annotate(r'$F_n(x)$', xytext=(3, 0.2), xycoords='data', xy=(xi[4], inv_n[3]), textcoords='data',
             fontsize=font_size,  va="baseline", ha="right",
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.4", facecolor='black', relpos=(0, 0.7),
                             shrinkA=0, shrinkB=1))

xa = 2.8
plt.annotate(r'$F(x)$', xytext=(4.2, 0.7), xycoords='data', xy=(xa, norm.cdf(xa, loc=mean, scale=variance)),
             textcoords='data', fontsize=font_size,  va="baseline", ha="right", color=grey,
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.4", relpos=(0.1, 0.9),
                             shrinkA=0, shrinkB=1, color=grey))

plt.axis('off')


ax = plt.subplot2grid((1, 10), (0, 5), rowspan=1, colspan=5)
plt.axis([y_ax_min, y_ax_max, x_ax_min, x_ax_max])
ax.set_aspect(abs(y_ax_max-y_ax_min)/abs(x_ax_max-x_ax_min))

# axis arrows
plt.annotate("", xytext=(y_ax_min, 0), xycoords='data', xy=(y_ax_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=5, headlength=7, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, x_ax_min), xycoords='data', xy=(0, x_ax_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=5, headlength=7, facecolor='black', shrink=0.002))

# axis labels
plt.text(y_ax_max, x_bl2, r'$u$', fontsize=font_size, ha='right', va='baseline')

plt.plot(q, invFx, color=grey, lw=2)
plt.plot(inv_n, xi, 'k.', markersize=6)

plt.plot([inv_n_ext, inv_n_ext], [xi_ext[:-1], xi_ext[1:]], 'k')
plt.plot([inv_n_ext[:-1], inv_n_ext[1:]], [xi, xi], 'k')
plt.plot([inv_n, inv_n], [np.zeros((n,)), xi], 'k')

# ticks
plt.plot([0, vtl], [xi, xi], 'k')

plt.text(y_rm2, xi[0], r'$x_{\mathrm{min}}$', fontsize=font_size, ha='right', va='center')
plt.text(y_rm2, xi[-1], r'$x_{\mathrm{max}}$', fontsize=font_size, ha='right', va='center')


plt.text(y_rm2, xi[i1], r'$x_i$', fontsize=font_size, ha='right', va='center')

plt.text(1, x_bl2, r'$1$', fontsize=font_size, ha='center', va='baseline')
plt.text(y_rm2, x_bl2, r'$0$', fontsize=font_size, ha='right', va='baseline')

plt.annotate(r'$F^{-1}(u)$', xytext=(0.7, 3.5), xycoords='data', xy=(norm.cdf(xa, loc=mean, scale=variance), xa),
             textcoords='data', fontsize=font_size,  va="baseline", ha="right", color=grey,
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.4", relpos=(0.6, 0.9),
                             shrinkA=0, shrinkB=1, color=grey))

# 1/n
plt.plot([inv_n[i2], inv_n[i2]], [yd-2*vtl, yd+2*vtl], 'k')
plt.plot([inv_n[i2+1], inv_n[i2+1]], [yd-2*vtl, yd+2*vtl], 'k')
plt.plot([inv_n[i2], inv_n[i2+1]], [yd, yd], 'k')
plt.text((inv_n[i2]+inv_n[i2+1])/2+0.01, yd+y_rm1, r'$\frac{1}{n}$', fontsize=font_size, ha='center', va='top')

plt.axis('off')
plt.savefig('percentiles_frequency_interpretation.pdf', bbox_inches='tight')

plt.show()
