import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)

z = 7

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

fig = plt.figure(1, figsize=(8, 8), frameon=False)
#
# REGION 1
#
ax = plt.subplot2grid((2, 4), (0, 0), rowspan=1, colspan=2)
plt.axis([ax_min, ax_max, ax_min, ax_max])
ax.set_aspect('equal', adjustable='box')


# axis arrows
plt.annotate("", xytext=(ax_min, 0), xycoords='data', xy=(ax_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ax_min), xycoords='data', xy=(0, ax_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# axis labels
plt.text(ax_max, -1.8, r'$x$', fontsize=font_size, ha='center', va='baseline')
plt.text(-0.6, ax_max, r'$y$', fontsize=font_size, ha='right', va='center')

# region 1
# region limit (x=y)
plt.plot([t_min, t_max], [t_min, t_max], 'k')
# filled region (x<y)
vertices = np.array([[t_max, t_max], [t_min, t_min], [t_max, t_min]])
ax.add_patch(Polygon(vertices, facecolor='#0343df', alpha=0.4, edgecolor='none'))

# region 2
# region limit (x=z)
plt.plot([z, z], [t_min, t_max], 'k')
# filled region (x<y)
vertices = np.array([[t_min, t_max], [z, t_max], [z, t_min], [t_min, t_min]])
ax.add_patch(Polygon(vertices, facecolor='#ff000d', alpha=0.4, edgecolor='none'))


# labels
plt.text(t_min/2, t_max/2, r'$x\leq z$', fontsize=font_size, ha='center', va='center')
plt.text(z, t_max, r'$x=z$', fontsize=font_size, ha='center', va='bottom')

plt.text(t_max-3, t_max-4, r'$x=y$', fontsize=font_size, ha='center', va='bottom', rotation=45)
plt.text((z+t_max)/2, t_min/2, r'$x>y$', fontsize=font_size, ha='center', va='center')

plt.text(z+0.4, z-0.7, r'$(z,\,z)$', fontsize=font_size, ha='left', va='center')
plt.plot(z, z, 'k.', markersize=8)

plt.text((t_min+t_max)/2, ax_min-3.5, r'$P\{\mathbf{x}\leq z,\,\mathbf{x}>\mathbf{y}\}$',
         fontsize=font_size, ha='center', va='baseline')

plt.axis('off')

#
# REGION 2
#
ax = plt.subplot2grid((2, 4), (0, 2), rowspan=1, colspan=2)
plt.axis([ax_min, ax_max, ax_min, ax_max])
ax.set_aspect('equal', adjustable='box')


# axis arrows
plt.annotate("", xytext=(ax_min, 0), xycoords='data', xy=(ax_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ax_min), xycoords='data', xy=(0, ax_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# axis labels
plt.text(ax_max, -1.8, r'$x$', fontsize=font_size, ha='center', va='baseline')
plt.text(-0.6, ax_max, r'$y$', fontsize=font_size, ha='right', va='center')

# region 1
# region limit (x=y)
plt.plot([t_min, t_max], [t_min, t_max], 'k')
# filled region (x<y)
vertices = np.array([[t_max, t_max], [t_min, t_min], [t_min, t_max]])
ax.add_patch(Polygon(vertices, facecolor='#0343df', alpha=0.4, edgecolor='none'))

# region 2
# region limit (x=z)
plt.plot([t_min, t_max], [z, z], 'k')
# filled region (x<y)
vertices = np.array([[t_max, t_min], [t_max, z], [t_min, z], [t_min, t_min]])
ax.add_patch(Polygon(vertices, facecolor='#ff000d', alpha=0.4, edgecolor='none'))


# labels
plt.text(t_min/2, (z+t_max)/2, r'$x\leq y$', fontsize=font_size, ha='center', va='center')
plt.text(t_max, t_max/2, r'$y=z$', fontsize=font_size, ha='center', va='center')

plt.text(t_max, t_max-4, r'$x=y$', fontsize=font_size, ha='center', va='bottom', rotation=45)
plt.text(t_max/2, t_min/2, r'$y\leq z$', fontsize=font_size, ha='center', va='center')

plt.text(z-0.4, z-1.6, r'$(z,\,z)$', fontsize=font_size, ha='left', va='center')
plt.plot(z, z, 'k.', markersize=8)

plt.text((t_min+t_max)/2, ax_min-3.5, r'$P\{\mathbf{y}\leq z,\,\mathbf{x}\leq \mathbf{y}\}$',
         fontsize=font_size, ha='center', va='baseline')

plt.axis('off')

#
# REGION UNION
#
ax = plt.subplot2grid((2, 4), (1, 1), rowspan=1, colspan=2)
plt.axis([ax_min, ax_max, ax_min, ax_max])
ax.set_aspect('equal', adjustable='box')


# axis arrows
plt.annotate("", xytext=(ax_min, 0), xycoords='data', xy=(ax_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ax_min), xycoords='data', xy=(0, ax_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# axis labels
plt.text(ax_max, -1.8, r'$x$', fontsize=font_size, ha='center', va='baseline')
plt.text(-0.6, ax_max, r'$y$', fontsize=font_size, ha='right', va='center')

# region limit
plt.plot([t_min, z], [t_min, z], 'k')
plt.plot([z, z], [t_min, z], 'k')
plt.plot([t_min, z], [z, z], 'k')
# filled region (x<y)
vertices = np.array([[t_min, z], [z, z], [z, t_min], [t_min, t_min]])
ax.add_patch(Polygon(vertices, facecolor='#0343df', alpha=0.4, edgecolor='none'))
vertices = np.array([[t_min, z], [z, z], [z, t_min], [t_min, t_min]])
ax.add_patch(Polygon(vertices, facecolor='#ff000d', alpha=0.4, edgecolor='none'))

# labels
plt.text(t_min/2, z/2, r'$x\leq z,$' '\n' r'$y\leq z$', fontsize=font_size, ha='center', va='center',
         multialignment='left')


plt.text(z, z, r'$(z,\,z)$', fontsize=font_size, ha='left', va='bottom')
plt.plot(z, z, 'k.', markersize=8)

plt.axis('off')

plt.text((t_min+t_max)/2, ax_min-3.5,
         r'$P\{(\mathbf{x}\leq z,\,\mathbf{x}>\mathbf{y})\cup(\mathbf{y}\leq z,\,\mathbf{x}\leq \mathbf{y})\}'
         r'=P\{\mathbf{x}\leq z,\,\mathbf{y}\leq z\}$', fontsize=font_size, ha='center', va='baseline')

# legend
handles = []
handle1 = Rectangle((0, 0), 1, 1, color='#0343df', alpha=0.4, linewidth=0)
handle2 = Rectangle((0, 0), 1, 1, color='#ff000d', alpha=0.4, linewidth=0)
handles.append(handle1)
handles.append(handle2)
handles.append((handle1, handle2))

plt.legend(handles=handles, labels=["Región 1", "Región 2", "Intersección"], framealpha=1, frameon=False,
           loc='center right', bbox_to_anchor=(0.18, 0.85))

plt.savefig('joint_distribution_region_rv_max.pdf', bbox_inches='tight')

plt.show()

