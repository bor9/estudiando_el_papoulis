import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)

x = 7
y = 5

# axis maximum
t_max = 12
delta_t = 0.5
ax_max = t_max + delta_t
# y tick margin
ytm = 0.6
# font size
font_size = 16

fig = plt.figure(1, figsize=(4, 4), frameon=False)
ax = fig.add_subplot(111)
plt.ylim(-ax_max, ax_max)
plt.xlim(-ax_max, ax_max)

# axis arrows
plt.annotate("", xytext=(-ax_max, 0), xycoords='data', xy=(ax_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, -ax_max), xycoords='data', xy=(0, ax_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# axis labels
plt.text(x+0.3, -0.8, r'$x$', fontsize=font_size, ha='left', va='center')
plt.text(-0.3, y+0.2, r'$y$', fontsize=font_size, ha='right', va='bottom')

# region limits
plt.plot([-t_max, x], [y, y], 'k')
plt.plot([x, x], [-t_max, y], 'k')

plt.plot(x, 0, 'k.', markersize=10)
plt.plot(0, y, 'k.', markersize=10)

plt.text(-4, -4, r'$D$', fontsize=font_size, ha='center', va='center')

# filled area with a polygon patch
# vertices
vertices = np.array([[x, y], [-t_max, y], [-t_max, -t_max], [x, -t_max]])
ax.add_patch(Polygon(vertices, color='#EEEEEE'))

plt.axis('off')
plt.savefig('joint_distribution_region.pdf', bbox_inches='tight')
plt.show()
