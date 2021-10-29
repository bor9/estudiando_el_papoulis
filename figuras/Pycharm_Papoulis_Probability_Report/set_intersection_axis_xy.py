import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)

x1 = 9
x2 = 17
y1 = 5
y2 = 12

lw = 0.15

# axis limits
t_min = -3
t_max = 24+t_min
delta_t = 1.5
ax_max = t_max + delta_t
ax_min = t_min - delta_t
# y tick margin
ytm = 0.6
# font size
font_size = 15
# legend background color
bggrey = 0.97

fig = plt.figure(0, figsize=(4, 4), frameon=False)

#
# w < z
#
ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
plt.axis([ax_min, ax_max, ax_min, ax_max])
ax.set_aspect('equal', adjustable='box')
# axis arrows
plt.annotate("", xytext=(ax_min, 0), xycoords='data', xy=(ax_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ax_min), xycoords='data', xy=(0, ax_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

#
# REGION 1
#
# region limit
plt.plot([x1, x1], [t_min, t_max], 'k')
plt.plot([x2, x2], [t_min, t_max], 'k')
# filled region (x<y)
vertices = np.array([[x1, t_min], [x2, t_min], [x2, t_max], [x1, t_max]])
ax.add_patch(Polygon(vertices, facecolor='#0343df', alpha=0.4, edgecolor='none'))

# line in the axis
vertices = np.array([[x1, -lw], [x2, -lw], [x2, lw], [x1, lw]])
ax.add_patch(Polygon(vertices, facecolor='k', edgecolor='k'))

#
# REGION 2
#
# region limit
plt.plot([t_min, t_max], [y1, y1], 'k')
plt.plot([t_min, t_max], [y2, y2], 'k')
# filled region (x<y)
vertices = np.array([[t_min, y1], [t_max, y1], [t_max, y2], [t_min, y2]])
ax.add_patch(Polygon(vertices, facecolor='#ff000d', alpha=0.4, edgecolor='none'))

# line in the axis
vertices = np.array([[-lw, y1], [lw, y1], [lw, y2], [-lw, y2]])
ax.add_patch(Polygon(vertices, facecolor='k', edgecolor='k'))

x_bl = -1.8
y_rm = -0.6
# axis labels
plt.text(ax_max, x_bl, r'$x$', fontsize=font_size, ha='center', va='baseline')
plt.text(y_rm, ax_max, r'$y$', fontsize=font_size, ha='right', va='center')

# x labels
plt.text(x1, x_bl, r'$x_1$', fontsize=font_size, ha='right', va='baseline')
plt.text(x2+0.2, x_bl, r'$x_2$', fontsize=font_size, ha='left', va='baseline')
plt.text((x1+x2)/2, x_bl-0.2, r'$A$', fontsize=font_size, ha='center', va='baseline')
# y labels
plt.text(y_rm+0.3, y1, r'$y_1$', fontsize=font_size, ha='right', va='top')
plt.text(y_rm+0.3, y2+0.2, r'$y_2$', fontsize=font_size, ha='right', va='bottom')
plt.text(y_rm, (y1 + y2)/2, r'$B$', fontsize=font_size, ha='right', va='center')

# set labels
plt.text(x1/2, (y1 + y2)/2, r'$S_1\times B$', fontsize=font_size, ha='center', va='center')
plt.text((x1+x2)/2, (y2 + t_max)/2, r'$A\times S_2$', fontsize=font_size, ha='center', va='center')
plt.text((x1+x2)/2, (y1 + y2)/2, r'$A\times B$', fontsize=font_size, ha='center', va='center')

plt.axis('off')

plt.savefig('set_intersection_axis_xy.pdf', bbox_inches='tight')
plt.show()

