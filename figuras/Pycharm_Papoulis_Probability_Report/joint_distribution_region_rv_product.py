import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import Polygon
from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)

x_stripe = 3
y_stripe = 3
delta_stripe = 0.7
z = 3/2

# axis limits
t_min = -8
t_max = 24+t_min
delta_t = 1.5
ax_max = t_max + delta_t
ax_min = t_min - delta_t
# y tick margin
ytm = 0.6
# font size
font_size = 16

fig = plt.figure(1, figsize=(8, 4), frameon=False)
#
# HORIZONTAL STRIPE
#
ax = fig.add_subplot(121)
plt.axis('equal')
plt.axis([ax_min, ax_max, ax_min, ax_max])

# axis arrows
plt.annotate("", xytext=(ax_min, 0), xycoords='data', xy=(ax_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ax_min), xycoords='data', xy=(0, ax_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# axis labels
plt.text(ax_max, -2, r'$x$', fontsize=font_size, ha='center', va='baseline')
plt.text(-1, ax_max, r'$y$', fontsize=font_size, ha='right', va='center')

# region limit
plt.plot([t_min, t_max], [t_min/z, t_max/z], 'k')

# filled area with a polygon patch
# vertices
vertices = np.array([[0, 0], [t_max, t_max/z], [t_max, t_max], [t_min, t_max], [t_min, 0]])
ax.add_patch(Polygon(vertices, color='#EEEEEE'))

# filled area with a polygon patch
vertices = np.array([[t_min, y_stripe], [z*y_stripe, y_stripe], [z*(y_stripe+delta_stripe), y_stripe+delta_stripe],
                     [t_min, y_stripe+delta_stripe]])
ax.add_patch(Polygon(vertices, color='#AAAAAA'))
# stripe limits
plt.plot([t_min, z*y_stripe], [y_stripe, y_stripe], 'k')
plt.plot([t_min, z*(y_stripe+delta_stripe)], [y_stripe+delta_stripe, y_stripe+delta_stripe], 'k')

# labels
plt.text(z*y_stripe+1.7, y_stripe-0.2, r'$x=yz$', fontsize=font_size, ha='left', va='baseline')
plt.text(t_max-0.5, t_max-2, r'$x/y\leq z,\;\; y>0$', fontsize=font_size, ha='right', va='baseline')


plt.text(-5, -7, r'$x=yz$', fontsize=font_size, ha='center', va='bottom', rotation=math.degrees(math.atan(1/z)))
plt.axis('off')

#
# VERTICAL STRIPE
#
# axis limits
t_max = 8
t_min = -24 + t_max
ax_max = t_max + delta_t
ax_min = t_min - delta_t

ax = fig.add_subplot(122)
plt.axis('equal')
plt.axis([ax_min, ax_max, ax_min, ax_max])

# axis arrows
plt.annotate("", xytext=(ax_min, 0), xycoords='data', xy=(ax_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ax_min), xycoords='data', xy=(0, ax_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# axis labels
plt.text(ax_max, -2, r'$x$', fontsize=font_size, ha='center', va='baseline')
plt.text(-1, ax_max, r'$y$', fontsize=font_size, ha='right', va='center')

# region limit
plt.plot([t_min, t_max], [t_min/z, t_max/z], 'k')

# filled area with a polygon patch
# vertices
vertices = np.array([[t_min, t_min], [t_max, t_min], [t_max, 0], [0, 0], [t_min, t_min/z]])
ax.add_patch(Polygon(vertices, color='#EEEEEE'))

y_stripe = -y_stripe
# filled area with a polygon patch
vertices = np.array([[z*y_stripe, y_stripe], [z*(y_stripe+delta_stripe), y_stripe+delta_stripe],
                     [t_max, y_stripe+delta_stripe], [t_max, y_stripe]])
ax.add_patch(Polygon(vertices, color='#AAAAAA'))
# stripe limits
plt.plot([z*y_stripe, t_max], [y_stripe, y_stripe], 'k')
plt.plot([z*(y_stripe+delta_stripe), t_max], [y_stripe+delta_stripe, y_stripe+delta_stripe], 'k')

# labels
plt.text(z*y_stripe-1, y_stripe-0.2, r'$x=yz$', fontsize=font_size, ha='right', va='baseline')
plt.text(t_min+0.5, t_min+1, r'$x/y\leq z,\;\; y<0$', fontsize=font_size, ha='left', va='baseline')

plt.text(5.3, 2.5, r'$x=yz$', fontsize=font_size, ha='center', va='bottom', rotation=math.degrees(math.atan(1/z)))
plt.axis('off')

plt.savefig('joint_distribution_region_rv_product.pdf', bbox_inches='tight')
plt.show()

