import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import patches
import matplotlib.colors as colors

from matplotlib import cm
from matplotlib import rc


__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=True)
rc('mathtext', fontset='cm')

# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col10 = scalarMap.to_rgba(0)

# read image
im = 'buffon_needle_3.png'
img = plt.imread(im)

# image parameters
img_height = 25
img_width = img_height * img.shape[1] / img.shape[0]

# axis span
x_max = 35
y_max = 42

# needle image position
x_start = 13
y_start = 1

# lines y coordinate
r1 = y_start
r2 = 41

# compute needle angle
theta = math.atan(img.shape[0]/img.shape[1])

# needle center
xc = x_start + img_width / 2
yc = y_start + img_height / 2

# needle size mark
r = 5  # distance from the needle
s = 1  # lines of the borders

# parameters for plot
dashes = (5, 2)
fontsize = 14

# plot
fig = plt.figure(1, figsize=(5, 5), frameon=False)
ax = fig.add_subplot(111)
ax.set_xlim(0, x_max)
ax.set_ylim(0, y_max)

# plot lines
plt.plot([0, x_max], [r1, r1], color=col10, lw=2)
plt.plot([0, x_max], [r2, r2], color=col10, lw=2)

# plot angle
e = patches.Arc((x_start, y_start), 8, 8, angle=0, theta1=0, theta2=theta * 180 / math.pi, linewidth=1,
                fill=False, zorder=1)
ax.add_patch(e)
plt.text(x_start + 5.4 * math.cos(theta/2), y_start + 5.4 * math.sin(theta/2), r'$\theta$', fontsize=fontsize,
         ha='center', va='center')

# show needle image
ax.imshow(img, extent=[x_start, x_start + img_width, y_start, y_start + img_height], zorder=2)

# mark needle center
plt.plot(xc, yc, 'k.', markersize=6, zorder=3)
plt.plot([xc, xc], [r1, yc], 'k--', lw=1, dashes=dashes)
plt.text(xc + 1, (r1 + yc) / 2, r'$$z=\frac{l}{2}\sin\theta$$', fontsize=fontsize, ha='left', va='center')

# plot needle size marker
xm = x_start - r * math.sin(theta)
ym = y_start + r * math.cos(theta)
plt.plot(xm + np.array([0, img_width]), ym + np.array([0, img_height]), 'k--', lw=1, dashes=dashes)
plt.plot(xm + np.array([-s*math.sin(theta), s*math.sin(theta)]),
         ym + np.array([s*math.cos(theta), -s*math.cos(theta)]),
         'k-', lw=1)
plt.plot(xm + img_width + np.array([-s*math.sin(theta), s*math.sin(theta)]),
         ym + img_height + np.array([s*math.cos(theta), -s*math.cos(theta)]),
         'k-', lw=1)
plt.text(xm + img_width / 2 - 2 * math.sin(theta), ym + img_height / 2 + 2 * math.cos(theta), r'$l$',
         fontsize=fontsize, ha='center', va='baseline')

# plot distance between lines
xl = 2
plt.plot([xl, xl], [y_start, r2], 'k--', lw=1, dashes=dashes)
plt.plot(xl + np.array([-s, s]), r1 + np.array([-0.5, 0.5]), 'k-', lw=1)
plt.plot(xl + np.array([-s, s]), r2 + np.array([-0.5, 0.5]), 'k-', lw=1)
plt.text(xl + 1, (r1 + r2) / 2, r'$d$', fontsize=fontsize, ha='left', va='center')

plt.axis('off')
plt.savefig('buffon_needle_short.pdf', bbox_inches='tight', dpi=900)
plt.show()
