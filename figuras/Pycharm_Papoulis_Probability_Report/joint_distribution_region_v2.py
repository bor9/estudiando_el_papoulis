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
# x and y tick margin
x_baseline = -2
x_side = 0.3
y_side = -0.5
y_vert = 0.2
# font size
font_size = 14

fig = plt.figure(1, figsize=(8, 4), frameon=False)
ax = plt.subplot2grid((1, 6), (0, 0), rowspan=1, colspan=2)
plt.axis([-ax_max, ax_max, -ax_max, ax_max])
ax.set_aspect('equal', adjustable='box')

# axis arrows
plt.annotate("", xytext=(-ax_max, 0), xycoords='data', xy=(ax_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=4, headlength=6, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, -ax_max), xycoords='data', xy=(0, ax_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=4, headlength=6, facecolor='black', shrink=0.002))

# axis labels
plt.text(x+x_side, x_baseline, r'$x$', fontsize=font_size, ha='left', va='baseline')
plt.text(y_side, y+y_vert, r'$y$', fontsize=font_size, ha='right', va='bottom')

# region limits
plt.plot([-t_max, x], [y, y], 'k')
plt.plot([x, x], [-t_max, y], 'k')

plt.plot(x, 0, 'k.', markersize=8)
plt.plot(0, y, 'k.', markersize=8)

plt.text(-t_max/2, -t_max/2, r'$D_1$', fontsize=font_size, ha='center', va='center')

# filled area with a polygon patch
# vertices
vertices = np.array([[x, y], [-t_max, y], [-t_max, -t_max], [x, -t_max]])
ax.add_patch(Polygon(vertices, color='#EEEEEE'))

plt.axis('off')

# REGION: (x1 < X < x2, Y < y)
# axis limits
y_max = 12
y_min = -y_max
x_min = -6
x_max = 2 * y_max + x_min

ay_max = y_max + delta_t
ay_min = -ay_max
ax_max = x_max + delta_t
ax_min = x_min - delta_t

y1 = 3
y2 = 9
yx = 5

x1 = 9
x2 = x1 + (y2 - y1)
xy = 5

ax = plt.subplot2grid((1, 6), (0, 2), rowspan=1, colspan=2)
plt.axis([ax_min, ax_max, ay_min, ay_max])
ax.set_aspect('equal', adjustable='box')

# axis arrows
plt.annotate("", xytext=(ax_min, 0), xycoords='data', xy=(ax_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=4, headlength=6, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ay_min), xycoords='data', xy=(0, ay_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=4, headlength=6, facecolor='black', shrink=0.002))

# region x1 < x < x2
plt.plot([x1, x1], [y_min, xy], 'k')
plt.plot([x2, x2], [y_min, xy], 'k')
plt.plot([x1, x2], [xy, xy], 'k')

plt.plot(x1, 0, 'k.', markersize=8)
plt.plot(x2, 0, 'k.', markersize=8)
plt.plot(0, xy, 'k.', markersize=8)

# filled area with a polygon patch
vertices = np.array([[x1, y_min], [x1, xy], [x2, xy], [x2, y_min]])
ax.add_patch(Polygon(vertices, color='#EEEEEE'))

# labels
plt.text(x1, x_baseline, r'$x_1$', fontsize=font_size, ha='right', va='baseline')
plt.text(x2+x_side, x_baseline, r'$x_2$', fontsize=font_size, ha='left', va='baseline')
plt.text(xy, x_baseline, r'$x$', fontsize=font_size, ha='center', va='baseline')
plt.text((x1+x2)/2, y_min/2, r'$D_2$', fontsize=font_size, ha='center', va='center')


# region y1 < y < y2
plt.plot([x_min, yx], [y1, y1], 'k')
plt.plot([x_min, yx], [y2, y2], 'k')
plt.plot([yx, yx], [y1, y2], 'k')

plt.plot(0, y1, 'k.', markersize=8)
plt.plot(0, y2, 'k.', markersize=8)
plt.plot(yx, 0, 'k.', markersize=8)

# filled area with a polygon patch
vertices = np.array([[x_min, y2], [yx, y2], [yx, y1], [x_min, y1]])
ax.add_patch(Polygon(vertices, color='#EEEEEE'))

# labels
plt.text(0, y1+y_vert, r'$y_1$', fontsize=font_size, ha='right', va='top')
plt.text(0, y2+y_vert, r'$y_2$', fontsize=font_size, ha='right', va='bottom')
plt.text(y_side, xy, r'$y$', fontsize=font_size, ha='right', va='center')
plt.text(x_min+0.2, (y1+y2)/2, r'$D_3$', fontsize=font_size, ha='left', va='center')


plt.axis('off')

# REGION: (x1 < X < x2, y1 < Y < y2)
ax = plt.subplot2grid((1, 6), (0, 4), rowspan=1, colspan=2)
plt.axis([ax_min, ax_max, ay_min, ay_max])
ax.set_aspect('equal', adjustable='box')

# axis arrows
plt.annotate("", xytext=(ax_min, 0), xycoords='data', xy=(ax_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=4, headlength=6, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ay_min), xycoords='data', xy=(0, ay_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=4, headlength=6, facecolor='black', shrink=0.002))

# region
plt.plot([yx, yx], [y1, y2], 'k')
plt.plot([yx, x2], [y1, y1], 'k')
plt.plot([x2, x2], [y1, y2], 'k')
plt.plot([xy, x2], [y2, y2], 'k')

plt.plot(yx, 0, 'k.', markersize=8)
plt.plot(x2, 0, 'k.', markersize=8)
plt.plot(0, y1, 'k.', markersize=8)
plt.plot(0, y2, 'k.', markersize=8)

# filled area with a polygon patch
vertices = np.array([[xy, y2], [xy, y1], [x2, y1], [x2, y2]])
ax.add_patch(Polygon(vertices, color='#EEEEEE'))

# labels
plt.text(x2+x_side, x_baseline, r'$x_2$', fontsize=font_size, ha='center', va='baseline')
plt.text(xy, x_baseline, r'$x_1$', fontsize=font_size, ha='center', va='baseline')
plt.text(y_side, y1, r'$y_1$', fontsize=font_size, ha='right', va='center')
plt.text(y_side, y2, r'$y_2$', fontsize=font_size, ha='right', va='center')
plt.text((xy+x2)/2, (y1+y2)/2, r'$D_4$', fontsize=font_size, ha='center', va='center')

plt.axis('off')






plt.savefig('joint_distribution_region_v2.pdf', bbox_inches='tight')
plt.show()
