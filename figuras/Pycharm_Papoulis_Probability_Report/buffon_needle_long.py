import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import patches
from matplotlib import transforms
import matplotlib.colors as colors

from matplotlib import cm
from matplotlib import rc


__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)
rc('mathtext', fontset='cm')

# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col10 = scalarMap.to_rgba(0)
col11 = scalarMap.to_rgba(0.2)
col20 = scalarMap.to_rgba(1)
col21 = scalarMap.to_rgba(0.85)
col22 = scalarMap.to_rgba(0.7)

# read image
im = 'buffon_needle_3.png'
img = plt.imread(im)

# image parameters
img_height = 25
img_width = img_height * img.shape[1] / img.shape[0]
# needle original angle (radians)
theta = math.atan(img.shape[0]/img.shape[1])
# needle length
l = math.sqrt(img_height**2 + img_width**2)

# axis span
x_max = 35
y_max = 42

# lines y-coordinate
n_lines = 3  # number of lines
start_gap = 1  # y-coordinate of the first lines
d = (y_max - 2 * start_gap) / (n_lines - 1)  # distance between lines
lines_y = start_gap + np.arange(n_lines) * d  # y-coordinate of lines

# needle 1 position
x1_start = 8
y1_start = lines_y[1]
# needle 1 angle
theta1 = math.asin(d/l)
rot1 = transforms.Affine2D().rotate_around(x1_start, y1_start, theta1 - theta)
# needle 2 height and width
img1_height = l * math.sin(theta1)
img1_width = l * math.cos(theta1)

# needle 2 position
x2_start = 3
y2_start = start_gap
# needle 2 angle
theta2 = theta1 * 0.6
rot2 = transforms.Affine2D().rotate_around(x2_start, y2_start, theta2 - theta)
# needle 2 height and width
img2_height = l * math.sin(theta2)
img2_width = l * math.cos(theta2)


# needle center
x2c = x2_start + img2_width / 2
y2c = y2_start + img2_height / 2

# needle size mark
r = 4  # distance from the needle
s = 1  # lines of the borders

# parameters for plot
dashes = (5, 2)
fontsize = 14

# plot
fig = plt.figure(0, figsize=(10, 5), frameon=False)
ax = plt.subplot2grid((1, 10), (0, 0), rowspan=1, colspan=5)
ax.set_xlim(0, x_max)
ax.set_ylim(0, y_max)

# plot lines
plt.plot([0, x_max], [lines_y, lines_y], color=col10, lw=2)

# plot angle arc 1
arc1 = patches.Arc((x1_start, y1_start), 10, 10, angle=0, theta1=0, theta2=theta1 * 180 / math.pi, linewidth=1,
                   fill=False, zorder=1)
ax.add_patch(arc1)
plt.text(x1_start + 6 * math.cos(theta1/2), y1_start + 6 * math.sin(theta1/2), r'$\theta=\arcsin\;\dfrac{d}{l}$',
         fontsize=fontsize, ha='left', va='center')

# plot angle arc 2
arc2 = patches.Arc((x2_start, y2_start), 10, 10, angle=0, theta1=0, theta2=theta2 * 180 / math.pi, linewidth=1,
                   fill=False, zorder=1)
ax.add_patch(arc2)
plt.text(x2_start + 6 * math.cos(theta2/2), y2_start + 6 * math.sin(theta2/2), r'$\theta$', fontsize=fontsize,
         ha='left', va='center')

# show needle 1 image
ax.imshow(img, transform=rot1 + ax.transData,
          extent=[x1_start, x1_start + img_width, y1_start, y1_start + img_height], zorder=2)

# show needle 2 image
ax.imshow(img, transform=rot2 + ax.transData,
          extent=[x2_start, x2_start + img_width, y2_start, y2_start + img_height], zorder=2)

# mark needle 1 center
plt.plot(x2c, y2c, 'k.', markersize=6, zorder=3)
plt.plot([x2c, x2c], [lines_y[0], y2c], 'k--', lw=1, dashes=dashes)
plt.text(x2c + 1, (lines_y[0] + y2c) / 2, r'$z=\dfrac{l}{2}\;\sin\;\theta$', fontsize=fontsize, ha='left', va='center')

# plot distance between lines
xl = 2
plt.plot((x1_start + img1_width) * np.array([1, 1]), lines_y[1] + np.array([0, img1_height]),
         'k--', lw=1, dashes=dashes)
plt.text(x1_start + img1_width + 1, lines_y[1] + img1_height / 2, r'$d$', fontsize=fontsize, ha='left', va='center')

# plot needle 2 size marker
xm = x2_start - r * math.sin(theta2)
ym = y2_start + r * math.cos(theta2)
plt.plot(xm + np.array([0, img2_width]), ym + np.array([0, img2_height]), 'k--', lw=1, dashes=dashes)
plt.plot(xm + s * math.sin(theta2) * np.array([-1, 1]),
         ym + s * math.cos(theta2) * np.array([1, -1]),
         'k-', lw=1)
plt.plot(xm + img2_width + s * math.sin(theta2) * np.array([-1, 1]),
         ym + img2_height + s * math.cos(theta2) * np.array([1, -1]),
         'k-', lw=1)
plt.text(xm + img2_width / 2 - 1 * math.sin(theta2), ym + img2_height / 2 + 1 * math.cos(theta2), r'$l$',
         fontsize=fontsize, ha='center', va='baseline')

plt.axis('off')

# SAMPLE SPACE PLOT

# scale l and d
f = 10
l /= f
d /= f

# axis limits
z_max = l / 2
t_max = math.pi / 2

delta_ax = 0.3
z_ax_min = -0.1
z_ax_max = z_max + delta_ax
t_ax_min = -0.1
t_ax_max = t_max + delta_ax

# theta vector
ts = np.linspace(0, t_max, 100)
sin_ts = (l / 2) * np.sin(ts)
zs = np.linspace(0, d/2, 100)
asin_zs = np.arcsin(2 * zs / l)

ax = plt.subplot2grid((1, 10), (0, 5), rowspan=1, colspan=5)
plt.axis([z_ax_min, z_ax_max, t_ax_min, t_ax_max])
# axis arrows
plt.annotate("", xytext=(z_ax_min, 0), xycoords='data', xy=(z_ax_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, t_ax_min), xycoords='data', xy=(0, t_ax_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot([d/2, d/2], [0, t_max], color=col10, lw=2)
plt.plot([0, d/2], [t_max, t_max], color=col10, lw=2)
plt.plot(sin_ts, ts, color=col20, lw=2)

ax.fill_between([0, d/2], math.asin(d/l) * np.array([1, 1]), t_max, color=col21)
ax.fill_between(zs, asin_zs, math.asin(d/l), color=col22)


# z labels
z_baseline = -0.14
plt.text(z_ax_max, z_baseline, r'$z$', fontsize=fontsize, ha='center', va='baseline')
plt.text(d/2, z_baseline, r'$\dfrac{d}{2}$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-0.05, z_baseline, r'$0$', fontsize=fontsize, ha='right', va='baseline')
plt.plot([l/2, l/2], [0, math.pi/2], 'k--', lw=1, dashes=dashes)
plt.text(l/2, z_baseline, r'$\dfrac{l}{2}$', fontsize=fontsize, ha='center', va='baseline')

# theta labels
plt.text(-0.05, t_ax_max, r'$\theta$', fontsize=fontsize, ha='right', va='center')
plt.text(-0.05, math.pi/2, r'$\dfrac{\pi}{2}$', fontsize=fontsize, ha='right', va='center')
plt.plot([0, math.pi/2], [l/2, l/2], 'k--', lw=1, dashes=dashes, zorder=1)
plt.plot([0, d/2], math.asin(d/l) * np.array([1, 1]), 'k--', lw=1, dashes=dashes)
plt.text(-0.05, math.asin(d/l), r'$\arcsin\;\dfrac{d}{l}$', fontsize=fontsize, ha='right', va='center')

plt.text(d/2, math.pi/2, r'$\Omega$', fontsize=fontsize, ha='right', va='bottom', color=col10)
z1 = 1.16
plt.annotate(r'$z=\dfrac{l}{2}\;\sin\;\theta$', xytext=((d+l)/4, 0.45), xycoords='data', xy=(z1, math.asin(2*z1/l)),
             textcoords='data', fontsize=fontsize, va="center", ha="center",
             arrowprops=dict(arrowstyle="-|>, head_width=0.1, head_length=0.4", facecolor='black', relpos=(0.4, 1),
                             patchA=None, patchB=None, shrinkA=4, shrinkB=1))

plt.text(d/4, (math.pi/2+math.asin(d/l)) / 2, r'$D_1$', fontsize=fontsize, ha='center', va='center')
plt.text(d/4, 0.75*math.asin(d/l), r'$D_2$', fontsize=fontsize, ha='center', va='center')


plt.axis('off')
plt.savefig('buffon_needle_long.pdf', bbox_inches='tight', dpi=900)
plt.show()
