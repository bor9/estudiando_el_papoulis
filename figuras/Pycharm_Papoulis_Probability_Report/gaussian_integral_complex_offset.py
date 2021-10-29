import matplotlib.pyplot as plt

from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)
rc('mathtext', fontset='cm')

#####################################
# PARAMETERS - This can be modified #
#####################################

# range of x of interest
xmin = -1
xmax = 1
ymin = -0.3
ymax = 1

T = 0.7
T_arrow = T/2
a = 0.6
b = 0.5

#####################
# END OF PARAMETERS #
#####################

# vertical tick margin
vtm = -0.13
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
plt.annotate("", xytext=(0, ymin), xycoords='data', xy=(0, ymax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# z = x
plt.arrow(-T, 0, T/2, 0, hold=None, width=0.01, length_includes_head=True, head_width=0.06, head_length=0.1,
          head_starts_at_zero=False, overhang=0.2, color='k')
plt.arrow(-T, 0, 3*T/2, 0, hold=None, width=0.01, length_includes_head=True, head_width=0.06, head_length=0.1,
          head_starts_at_zero=False, overhang=0.2, color='k')
plt.arrow(-T, 0, 2*T, 0, hold=None, width=0.01, length_includes_head=True, head_width=0, head_length=0,
          head_starts_at_zero=False, overhang=0.2, color='k')
# z = T + iy
plt.arrow(T, 0, 0, b, hold=None, width=0.01, length_includes_head=True, head_width=0, head_length=0,
          head_starts_at_zero=False, overhang=0.2, color='k')
plt.arrow(T, 0, 0, b/2, hold=None, width=0.01, length_includes_head=True, head_width=0.06, head_length=0.1,
          head_starts_at_zero=False, overhang=0.2, color='k')
# z = x + ib
plt.arrow(T, b, -2*T, 0, hold=None, width=0.01, length_includes_head=True, head_width=0, head_length=0,
          head_starts_at_zero=False, overhang=0.2, color='k')
plt.arrow(T, b, -T/2, 0, hold=None, width=0.01, length_includes_head=True, head_width=0.06, head_length=0.1,
          head_starts_at_zero=False, overhang=0.2, color='k')
plt.arrow(T, b, -3*T/2, 0, hold=None, width=0.01, length_includes_head=True, head_width=0.06, head_length=0.1,
          head_starts_at_zero=False, overhang=0.2, color='k')
# z = -T + iy
plt.arrow(-T, b, 0, -b, hold=None, width=0.01, length_includes_head=True, head_width=0, head_length=0,
          head_starts_at_zero=False, overhang=0.2, color='k')
plt.arrow(-T, b, 0, -b/2, hold=None, width=0.01, length_includes_head=True, head_width=0.06, head_length=0.1,
          head_starts_at_zero=False, overhang=0.2, color='k')

plt.plot(a, b, 'r.', markersize=13)

# xlabels and ylabels
plt.text(xmax, vtm, '$x$', fontsize=fontsize, ha='right', va='baseline')
plt.text(T, vtm, '$T$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-T, vtm, '$-T$', fontsize=fontsize, ha='center', va='baseline')

plt.text(htm, ymax + 0.02, '$y$', fontsize=fontsize, ha='right', va='top')

plt.text(a - 0.02, b + 0.06, '$c=a+ib$', fontsize=fontsize, ha='left', va='baseline')
plt.text(-T, b + 0.06, '$\Gamma_c(T)$', fontsize=fontsize, ha='left', va='baseline')


plt.axis('off')

# save as eps image
plt.savefig('gaussian_integral_complex_offset.pdf', bbox_inches='tight')
plt.show()


