import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)

#####################################
# PARAMETERS - This can be modified #
#####################################

# g(x): 3rd grade polynomial
xi = np.array([-3, -0.5, 2.5]) # roots
A = 0.2
offset = 5

# f(x): normal density
mean = 1
variance = 2
B = 15

# y value
y = offset
dy = 1

# range of x axis
xmin = -4
xmax = 4
N = 200

# y axis maximum value
ymax = 8
ymin = -1

# extension to the left and right of the range of interest
delta_x = 0.4

#####################
# END OF PARAMETERS #
#####################

x = np.linspace(xmin, xmax, N)

# polynomial coefficients
c = A * np.poly(xi)
# polynomial samples
g = np.polyval(c, x)+offset

# values of polynomial in y+dy
cdy = c
cdy[-1] -= dy
xidx = np.sort(np.roots(cdy))

# normal distribution samples
fx = B * norm.pdf(x, loc=mean, scale=variance)

x_ax_min = xmin - delta_x
x_ax_max = xmax + delta_x

# PLOT PARAMETERS
# font size
fontsize = 14
# dashes length/space
dashed = (4, 4)
# baseline
bl = -0.4

fig = plt.figure(0, figsize=(8, 5), frameon=False)
ax = fig.add_subplot(111)

plt.xlim(x_ax_min, x_ax_max)
plt.ylim(ymin, ymax)

# axis arrows
plt.annotate("", xytext=(x_ax_min, 0), xycoords='data', xy=(x_ax_max, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin), xycoords='data', xy=(0, ymax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(x, g, 'k', lw=2)
plt.plot(x, fx, 'k', lw=2)

plt.plot([xmin, xmax], [y, y], 'k--', linewidth=0.8, dashes=dashed)
plt.plot([xmin, xmax], [y+dy, y+dy], 'k--', linewidth=0.8, dashes=dashed)

plt.plot(xi, y*np.ones(xi.shape), 'k.', markersize=10)
plt.plot(xidx, (y+dy)*np.ones(xidx.shape), 'k.', markersize=10)
plt.plot(xi, y*np.zeros(xi.shape), 'k.', markersize=10)

for i in np.arange(xi.size):
    plt.plot([xi[i], xi[i]], [0, B * norm.pdf(xi[i], loc=mean, scale=variance)], 'k')
    plt.plot([xidx[i], xidx[i]], [0, B * norm.pdf(xidx[i], loc=mean, scale=variance)], 'k')
    plt.plot([xi[i], xi[i]], [B * norm.pdf(xi[i], loc=mean, scale=variance), y], 'k', dashes=dashed)
    plt.plot([xidx[i], xidx[i]], [B * norm.pdf(xidx[i], loc=mean, scale=variance), y+dy], 'k', dashes=dashed)
    xf = np.linspace(xi[i], xidx[i], 20)
    ax.fill_between(xf, 0, B * norm.pdf(xf, loc=mean, scale=variance), color="#eeeeee")
    plt.plot([xi[i], xi[i]], [-0.7, 0], 'k', dashes=dashed)
    plt.plot([xidx[i], xidx[i]], [-0.7, 0], 'k', dashes=dashed)
    plt.annotate("", xytext=(xi[i], -0.65), xycoords='data', xy=(xidx[i], -0.65),
                 textcoords='data', fontsize=fontsize, va="top", ha="left",
                 arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.5", facecolor='black',
                                 shrinkA=0, shrinkB=0))
    plt.text((xi[i] + xidx[i]) / 2, -1.2, '$dx_{}$'.format(i + 1), fontsize=fontsize, ha='center', va='baseline')


plt.text(xi[0], bl, '$x_1$', fontsize=fontsize, ha='right', va='baseline')
plt.text(xi[1]+0.06, bl, '$x_2$', fontsize=fontsize, ha='left', va='baseline')
plt.text(xi[2], bl, '$x_3$', fontsize=fontsize, ha='right', va='baseline')

plt.text(0.13, y, '$y$', fontsize=fontsize, backgroundcolor='w', ha='left', va='center')
plt.text(0.13, y+dy, '$y+dy$', fontsize=fontsize, backgroundcolor='w', ha='left', va='center')
plt.annotate("", xytext=(-3.8, y), xycoords='data', xy=(-3.8, y+dy),
                 textcoords='data', fontsize=fontsize, va="top", ha="left",
                 arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.5", facecolor='black',
                                 shrinkA=0, shrinkB=0))
plt.text(-3.9, (2*y+dy)/2, '$dy$', fontsize=fontsize, ha='right', va='center')

plt.text(3.1, 7.2, '$g(x)$', fontsize=fontsize, ha='left', va='center')
plt.text(3.7, 1.5, '$f_x(x)$', fontsize=fontsize, ha='left', va='center')

plt.text(x_ax_max, bl, '$x$', fontsize=fontsize, ha='center', va='baseline')
plt.text(0.08, bl, '$0$', fontsize=fontsize, ha='left', va='baseline')

plt.axis('off')
# save as eps image
plt.savefig('fy_as_function_of_fx.pdf', bbox_inches='tight')
plt.show()
