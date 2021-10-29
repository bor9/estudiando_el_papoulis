import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)
rc('mathtext', fontset='cm')


# range of x axis
xmin_ax = -1
xmax_ax = 100
# range of y axis
ymin_ax = 0
ymax_ax = 100

fontsize = 14
grey = '#888888'
markersize = 9

##############################
# regression line construction
# set of points approximation by east squares
# points abscissas and ordinates
xi = np.array([0, 2,   4,   7,   9.5, 11, 12])
yi = np.array([0, 2.8, 4, 5, 6, 9,  12])
xi = xi / xi[-1]
yi = yi / yi[-1]
N = 4
A = np.vander(xi, N)
At = A.transpose()
cc = (np.linalg.inv(At @ A) @ At) @ yi
##############################

fig = plt.figure(0, figsize=(10, 4), frameon=False)
ax = fig.add_subplot(111)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# rectangle coordinates and dimensions
rect_bottom = 15
rect_width = 15
rect_height = 70

# LEFT PLOT: mean square estimation of y by a constant

# rectangle: sample space
rect1_x = 0
p = patches.Rectangle((rect1_x, rect_bottom), rect_width, rect_height, fill=False, lw=2)
ax.add_patch(p)

# y axis
yaxis1_x = rect1_x + 1.7 * rect_width
plt.annotate("", xytext=(yaxis1_x, 2), xycoords='data', xy=(yaxis1_x, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# outcomes coordinates
outcomes1_x = rect1_x + rect_width / 2
outcomes1_y = rect_bottom + np.array([15, 37, 45, 60])
# outcomes plot
plt.plot([outcomes1_x] * len(outcomes1_y), outcomes1_y, 'k.', markersize=markersize)

# random variable y coordinates
print(outcomes1_y)
rv1_x = yaxis1_x
rv1_y = np.array([40, 50, 70, 90])
# random variable y plot
plt.plot([rv1_x] * len(rv1_y), rv1_y, 'k.', markersize=markersize)
# mean of y values
c = np.mean(rv1_y)
# c plot
plt.plot(rv1_x, c, 'r.', markersize=markersize, zorder=5)

# connection arrows
angleA = (-10, -20, 30, 30)
angleB = (20, 10, 0, 0)
for i in np.arange(len(outcomes1_y)):
    plt.annotate("", xytext=(outcomes1_x, outcomes1_y[i]), xycoords='data', xy=(rv1_x, rv1_y[i]), textcoords='data',
                 arrowprops=dict(width=0, headwidth=4, headlength=7, facecolor='black',  shrink=0.04, lw=1,
                                 connectionstyle="angle3,angleA={:d},angleB={:d}".format(angleA[i], angleB[i])))
    # labels
    plt.text(outcomes1_x - 1.5, outcomes1_y[i], r'$\zeta_{:d}$'.format(len(outcomes1_y) - i), fontsize=fontsize,
             ha='right', va='center')
    plt.text(rv1_x + 1.5, rv1_y[i], r'$\mathbf{{y}}(\zeta_{0:d})=y_{0:d}$'.format(len(outcomes1_y) - i),
             fontsize=fontsize, ha='left', va='center')
plt.text(rv1_x + 1.5, c - 1, r'$c=E\{\mathbf{y}\}$', fontsize=fontsize, ha='left', va='center')
plt.text(yaxis1_x - 1, ymax_ax, '$y$', fontsize=fontsize, ha='right', va='center')
plt.text(rect1_x + rect_width / 2, rect_bottom - 2, '$S$', fontsize=fontsize, ha='center', va='top')


# RIGHT PLOT: mean sqare estimation of y by another rv x

# rectangle: sample space
rect2_x = rect1_x + 3 * rect_width
r = patches.Rectangle((rect2_x, rect_bottom), rect_width, rect_height, fill=False, lw=2)
ax.add_patch(r)

# y axis
yaxis2_x = rect2_x + 1.7 * rect_width
plt.annotate("", xytext=(yaxis2_x, 2), xycoords='data', xy=(yaxis2_x, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(yaxis2_x, 2), xycoords='data', xy=(xmax_ax, 2), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# outcomes coordinates
outcomes2_x = rect2_x + rect_width / 2
outcomes2_y = rect_bottom + np.array([10, 20, 28])
# outcomes plot
plt.plot([outcomes2_x] * len(outcomes2_y), outcomes2_y, 'k.', markersize=markersize)
# ellipse plot
ellipse_center_y = (outcomes2_y[0]+outcomes2_y[-1]) / 2
e = patches.Ellipse((outcomes2_x, ellipse_center_y), rect_width / 1.5, 2 * (ellipse_center_y - outcomes2_y[0] + 7),
                    angle=0.0, fill=False, lw=1.5)
ax.add_patch(e)

# random variable y coordinates
rv2_x = yaxis2_x
rv2_y = np.array([45, 70, 80])
# random variable y plot
plt.plot([rv2_x] * len(rv2_y), rv2_y, 'k.', markersize=markersize)

# plot x
x = (yaxis2_x + xmax_ax) / 2
plt.plot(x, 2, 'k.', markersize=markersize)


# connection arrows
angleA1 = (10, 20, 20)
angleB1 = (45, 60, 60)
angleA2 = (-70, 0, 0)
angleB2 = (10, 150, 120)
for i in np.arange(len(outcomes2_y)):
    plt.annotate("", xytext=(outcomes2_x, outcomes2_y[i]), xycoords='data', xy=(rv2_x, rv2_y[i]), textcoords='data',
                 arrowprops=dict(width=0, headwidth=4, headlength=7, facecolor='black',  shrink=0.03, lw=1,
                                 connectionstyle="angle3,angleA={:d},angleB={:d}".format(angleA1[i], angleB1[i])))
    plt.annotate("", xytext=(outcomes2_x, outcomes2_y[i]), xycoords='data', xy=(x, 2), textcoords='data',
                 arrowprops=dict(width=0, headwidth=4, headlength=7, facecolor='black', shrink=0.02, lw=1,
                                 connectionstyle="angle3,angleA={:d},angleB={:d}".format(angleA2[i], angleB2[i])))
    plt.plot([yaxis2_x, x], [rv2_y[i], rv2_y[i]], 'k', lw=1, linestyle='--', dashes=(4, 2))
    plt.plot(x, rv2_y[i], 'kx', markersize=6)
    plt.text(outcomes2_x - 1, outcomes2_y[i], r'$\zeta_{:d}$'.format(len(outcomes2_y) - i), fontsize=fontsize,
             ha='right', va='center')
    # labels
plt.plot([x, x], [2, rv2_y[-1]], 'k', lw=1, linestyle='--', dashes=(4, 2))
plt.text(yaxis2_x - 1, ymax_ax, '$y$', fontsize=fontsize, ha='right', va='center')
plt.text(rect2_x + rect_width / 2, rect_bottom - 2, '$S$', fontsize=fontsize, ha='center', va='top')

plt.text(outcomes2_x, outcomes2_y[-1] + 14, r'$\{\mathbf{x}=x\}$', fontsize=fontsize, ha='center', va='center')
i = 2
plt.text(rv2_x + 1, rv2_y[i], r'$\mathbf{{y}}(\zeta_{0:d})=y_{0:d}$'.format(len(outcomes2_y) - i),
         fontsize=fontsize, ha='left', va='bottom')
i = 1
plt.text(rv2_x + 1, rv2_y[i], r'$\mathbf{{y}}(\zeta_{0:d})=y_{0:d}$'.format(len(outcomes2_y) - i),
         fontsize=fontsize, ha='left', va='bottom')
i = 0
plt.text(rv2_x + 1, rv2_y[i]-1, r'$\mathbf{{y}}(\zeta_{0:d})=y_{0:d}$'.format(len(outcomes2_y) - i),
         fontsize=fontsize, ha='left', va='top')

plt.text(x, 0.5, '$x$', fontsize=fontsize, ha='center', va='top')


# mean of y values
phi = np.mean(rv2_y)
# c plot
plt.plot(x, phi, 'r.', markersize=markersize, zorder=5)

# regression line plot
xx = np.linspace(0, 1, 100)
y = np.polyval(cc, xx)
xx = (xmax_ax - yaxis2_x - 5) * xx
y = y * (ymax_ax - 60)
i = 50
xx = (x - xx[i]) + xx
y = (phi - y[i]) + y
plt.plot(xx, y, 'k', lw=2)

plt.text(x + 1, phi - 3, r'$\varphi(x)=E\{\mathbf{y}|x\}$', fontsize=fontsize, ha='left', va='center')


plt.axis('off')
plt.savefig('mean_square_estimation_scheme.pdf', bbox_inches='tight')

plt.show()
