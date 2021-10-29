import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)

x_stripe = 3
y_stripe = 3
delta_stripe = 0.7
z = 11

# axis maximum
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
plt.text(ax_max, -2, r'$x$', fontsize=font_size, ha='right', va='baseline')
plt.text(-1, ax_max, r'$y$', fontsize=font_size, ha='right', va='top')

# region limit
plt.plot([z-t_max, t_max], [t_max, z-t_max], 'k')

# filled area with a polygon patch
# vertices
vertices = np.array([[t_min, t_max], [t_min, t_min], [t_max, t_min], [t_max, z-t_max], [z-t_max, t_max]])
ax.add_patch(Polygon(vertices, color='#EEEEEE'))

# filled area with a polygon patch
vertices = np.array([[t_min, y_stripe], [z-y_stripe, y_stripe], [z-y_stripe-delta_stripe, y_stripe+delta_stripe],
                     [t_min, y_stripe+delta_stripe]])
ax.add_patch(Polygon(vertices, color='#AAAAAA'))
# stripe limits
plt.plot([t_min, z-y_stripe], [y_stripe, y_stripe], 'k')
plt.plot([t_min, z-y_stripe-delta_stripe], [y_stripe+delta_stripe, y_stripe+delta_stripe], 'k')

# labels
plt.text(z-y_stripe+0.5, y_stripe, r'$x=z-y$', fontsize=font_size, ha='left', va='baseline')
plt.text(2, -5, r'$x+y\leq z$', fontsize=font_size, ha='left', va='baseline')

x = 4
plt.text(x, z-x-2, r'$x+y=z$', fontsize=font_size, ha='center', va='bottom', rotation=-45)
plt.axis('off')

#
# VERTICAL STRIPE
#
ax = fig.add_subplot(122)
plt.axis('equal')
plt.axis([ax_min, ax_max, ax_min, ax_max])

# axis arrows
plt.annotate("", xytext=(ax_min, 0), xycoords='data', xy=(ax_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ax_min), xycoords='data', xy=(0, ax_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# axis labels
plt.text(ax_max, -2, r'$x$', fontsize=font_size, ha='right', va='baseline')
plt.text(-1, ax_max, r'$y$', fontsize=font_size, ha='right', va='top')

# region limit
plt.plot([z-t_max, t_max], [t_max, z-t_max], 'k')

# filled area with a polygon patch
# vertices
vertices = np.array([[t_min, t_max], [t_min, t_min], [t_max, t_min], [t_max, z-t_max], [z-t_max, t_max]])
ax.add_patch(Polygon(vertices, color='#EEEEEE'))

# filled area with a polygon patch
vertices = np.array([[z-y_stripe-delta_stripe, t_min], [z-y_stripe, t_min], [z-y_stripe, y_stripe],
                     [z-y_stripe-delta_stripe, y_stripe+delta_stripe]])
ax.add_patch(Polygon(vertices, color='#AAAAAA'))
# stripe limits
plt.plot([z-y_stripe-delta_stripe, z-y_stripe-delta_stripe], [t_min, y_stripe+delta_stripe], 'k')
plt.plot([z-y_stripe, z-y_stripe], [t_min, y_stripe], 'k')

# labels
plt.text(z-y_stripe, y_stripe+0.5, r'$y=z-x$', fontsize=font_size, ha='left', va='baseline')
#plt.text(0, -5, r'$x+y\leq z$', fontsize=font_size, ha='left', va='baseline')

x = 4
plt.text(x, z-x-2, r'$x+y=z$', fontsize=font_size, ha='center', va='bottom', rotation=-45)

plt.axis('off')

plt.savefig('joint_distribution_region_rv_sum.pdf', bbox_inches='tight')
plt.show()


#
# SPECIAL CASE WITH X AND Y POSITIVE
#
fig = plt.figure(2, figsize=(8, 4), frameon=False)
ax = fig.add_subplot(121)

plt.axis([ax_min, ax_max, ax_min, ax_max])
ax.set_aspect('equal', adjustable='box')


plt.annotate("", xytext=(ax_min, 0), xycoords='data', xy=(ax_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ax_min), xycoords='data', xy=(0, ax_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# axis labels
plt.text(ax_max, -2, r'$x$', fontsize=font_size, ha='right', va='baseline')
plt.text(z, -2, r'$z$', fontsize=font_size, ha='center', va='baseline')
plt.text(-1, ax_max, r'$y$', fontsize=font_size, ha='right', va='top')
plt.text(-1, z, r'$z$', fontsize=font_size, ha='right', va='center')

plt.plot(z, 0, 'k.', markersize=7)
plt.plot(0, z, 'k.', markersize=7)

# region limit
plt.plot([z-t_max, t_max], [t_max, z-t_max], 'k')

# filled area with a polygon patch
# vertices
vertices = np.array([[0, 0], [z, 0], [0, z]])
ax.add_patch(Polygon(vertices, color='#EEEEEE'))

# filled area with a polygon patch
vertices = np.array([[0, y_stripe], [z-y_stripe, y_stripe], [z-y_stripe-delta_stripe, y_stripe+delta_stripe],
                     [0, y_stripe+delta_stripe]])
ax.add_patch(Polygon(vertices, color='#AAAAAA'))
# stripe limits
plt.plot([0, z-y_stripe], [y_stripe, y_stripe], 'k')
plt.plot([0, z-y_stripe-delta_stripe], [y_stripe+delta_stripe, y_stripe+delta_stripe], 'k')

# labels
plt.text(z-y_stripe+0.5, y_stripe, r'$x=z-y$', fontsize=font_size, ha='left', va='baseline')

x = 4
plt.text(x, z-x-2, r'$x+y=z$', fontsize=font_size, ha='center', va='bottom', rotation=-45)
plt.axis('off')

plt.savefig('joint_distribution_region_rv_sum_positive.pdf', bbox_inches='tight')
plt.show()
