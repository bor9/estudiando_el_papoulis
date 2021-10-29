import matplotlib.pyplot as plt
import numpy as np
import math

from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)


#####################################
# PARAMETERS - This can be modified #
#####################################

# beta distribution parameters
n = 10
k = 3

# range of x of interest
x0 = 0
x1 = 1

# extension to the left and right of the range of interest
delta_xi = 0.1
delta_xs = 0.3
delta_y = 0.4
# number of samples in an interval of length 0.1
npoints = 40

#####################
# END OF PARAMETERS #
#####################

# abscissa values
xinf = x0 - delta_xi
xsup = x1 + delta_xs
x = np.arange(xinf, xsup, 0.1 / npoints)

# index of 0
idx_0 = int(round((x0 - xinf) / 0.1 * npoints))
# index of 1
idx_1 = int(round((x1 - xinf) / 0.1 * npoints))

# uniform density
uniform_pdf = np.zeros(x.shape)
uniform_pdf[idx_0 : idx_1+1] = 1

# beta density
beta_pdf = np.zeros(x.shape)
beta_pdf[idx_0: idx_1+1] = math.factorial(n+1)/(math.factorial(n-k)*math.factorial(k))\
                            * np.power(x[idx_0: idx_1+1], k) * (np.power(1-x[idx_0: idx_1+1], n-k))

ind_max = np.argmax(beta_pdf)
print("argmax of beta density: {}".format(x[ind_max]))
print("k/n: {}".format(k/n))

# axis parameters
dx = 0.08
xmin = xinf - dx
xmax = xsup + dx

ymax = beta_pdf[ind_max] + delta_y
ymin = -delta_y

# vertical tick margin
vtm = -0.36
# horizontal tick margin
htm = -0.05
# font size
fontsize = 16
bggrey = 0.97
# dashes length/space
dashed = (4, 4)

fig = plt.figure(0, figsize=(5, 3), frameon=False)
ax = fig.add_subplot(111)

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)


# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(0, ymax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.plot(x, uniform_pdf, 'k', linewidth=2)
plt.plot(x, beta_pdf, 'r', linewidth=2)

# legend
leg = plt.legend(['$f_p(p)=U(0,\,1)$', r'$f_{p|A}(p|A)=\beta(n,\,k)$'], loc=(0.42, 0.7), fontsize=14)
leg.get_frame().set_facecolor(bggrey*np.ones((3,)))
leg.get_frame().set_edgecolor(bggrey*np.ones((3,)))
# xlabels and xtickslabels
plt.plot([k/n, k/n], [0, beta_pdf[ind_max]], 'k--', linewidth=0.8, dashes=dashed)
plt.text(xmax, vtm, '$p$', fontsize=fontsize, ha='right', va='baseline')
plt.text(0, vtm, '$0$', fontsize=fontsize, ha='center', va='baseline')
plt.text(1, vtm, '$1$', fontsize=fontsize, ha='center', va='baseline')
plt.text(k/n, vtm, r'$\frac{k}{n}$', fontsize=fontsize, ha='center', va='baseline')
# ylabels and ytickslabels
plt.text(htm, 1, '$1$', fontsize=fontsize, ha='right', va='center')

plt.axis('off')

# save as eps image
plt.savefig('flip_coin_posterior_probability.pdf', bbox_inches='tight')
plt.show()


