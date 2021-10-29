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

# n distribution parameters
var = 3  # sigma^2
r = 2.5

# range of x of interest
x0 = 0
x1 = 2*math.pi

# extension to the left and right of the range of interest
delta_xi = 0.5
delta_xs = 2
delta_y = 0.05
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
uniform_pdf[idx_0: idx_1+1] = 1/(2*math.pi)

# posterior density
post_pdf = np.zeros(x.shape)
post_pdf_nz = np.exp(-((r-x[idx_0: idx_1+1])**2) / (2*var))  # non zero part of the distribution
# numerical integration
# manual implementation of trapezoid rule
int_post_pdf_nz = (2 * np.sum(post_pdf_nz[1:-1]) + post_pdf_nz[0] + post_pdf_nz[-1]) * (0.1 / (2 * npoints))
# integration using python trapz - only for test the result of manual implementation
int_post_pdf_nz_2 = np.trapz(post_pdf_nz, x=None, dx=0.1/npoints)
post_pdf[idx_0: idx_1+1] = post_pdf_nz / int_post_pdf_nz  # pdf normalisation

print("manual implementation of integration: {}".format(int_post_pdf_nz))
print("integration using trapz: {}".format(int_post_pdf_nz_2))


# axis parameters
dx = 0.08
xmin = xinf - dx
xmax = xsup + dx
ymax = 1/int_post_pdf_nz + delta_y
ymin = 0

# x labels baseline
xbl = -0.035
# y labels right margin
yrm = -0.2
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
plt.annotate("", xytext=(0, ymin), xycoords='data', xy=(0, ymax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.plot(x, uniform_pdf, 'k', linewidth=2)
plt.plot(x, post_pdf, 'r', linewidth=2)

# legend
leg = plt.legend([r'$f_{\theta}(\theta)$', r'$f_{\theta\,|\,r}(\theta\,|\,r)$'], loc=(0.6, 0.7), fontsize=14)
leg.get_frame().set_facecolor(bggrey*np.ones((3,)))
leg.get_frame().set_edgecolor(bggrey*np.ones((3,)))
# xlabels and xtickslabels
plt.plot([r, r], [0, 1/int_post_pdf_nz], 'k--', linewidth=0.8, dashes=dashed)
plt.text(xmax, xbl, r'$\theta$', fontsize=fontsize, ha='right', va='baseline')
plt.text(0, xbl, '$0$', fontsize=fontsize, ha='center', va='baseline')
plt.text(r, xbl, r'$\theta=r$', fontsize=fontsize, ha='center', va='baseline')
plt.text(2*math.pi, xbl, r'$2\pi$', fontsize=fontsize, ha='center', va='baseline')
# ylabels and ytickslabels
plt.text(yrm, 1/(2*math.pi), r'$\frac{1}{2\pi}$', fontsize=22, ha='right', va='center')

plt.axis('off')

# save as eps image
plt.savefig('random_phase_posterior_probability.pdf', bbox_inches='tight')
plt.show()


