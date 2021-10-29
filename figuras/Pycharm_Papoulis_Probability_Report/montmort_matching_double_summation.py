import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.patches import FancyBboxPatch

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)

K = 6
t_max = K + 2
t_min = -1

# ticks length
tl = t_max/40
# y tick margin
ytm = 0.6
x_labels_baseline = -0.8
y_labels_center = -0.6
# font size
font_size1 = 16
font_size2 = 14

fig = plt.figure(1, figsize=(4, 4), frameon=False)
ax = fig.add_subplot(111)
plt.ylim(t_min, t_max)
plt.xlim(t_min, t_max)

# axis arrows
plt.annotate("", xytext=(t_min, 0), xycoords='data', xy=(t_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, t_min), xycoords='data', xy=(0, t_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# axis labels
plt.text(t_max, x_labels_baseline, r'$n$', fontsize=font_size1, ha='center', va='baseline')
plt.text(0.4, t_max, r'$k$', fontsize=font_size1, ha='left', va='top')


# rectangle of points
for i in np.arange(K+1):
    plt.plot(np.arange(i, K+1), i*np.ones(K-i+1), 'k.', markersize=10)
    p_fancy = FancyBboxPatch((i, 0), 0, i, boxstyle="round,pad=0.2", fc="none", ec=(0, .5, 0), zorder=4)
    ax.add_patch(p_fancy)
    p_fancy = FancyBboxPatch((i, i), K+1-i, 0, boxstyle="round,pad=0.2", fc="none", ec=(0.5, 0, 0), zorder=4)
    ax.add_patch(p_fancy)

# dots
for i in np.arange(1, K + 1):
    plt.plot(K+np.array([0.6, 0.8, 1]), i*np.ones(3), 'k.', markersize=2)
plt.plot(K+np.array([0.6, 0.8, 1]), -0.6*np.ones(3), 'k.', markersize=2)
plt.plot(K+np.array([0.6, 0.8, 1]), K+np.array([0.6, 0.8, 1]), 'k.', markersize=2)
plt.plot(y_labels_center*np.ones(3), K+np.array([0.6, 0.8, 1]), 'k.', markersize=2)

# axis labels and ticks
for i in np.arange(1, K+1):
    plt.text(i, x_labels_baseline, r'${}$'.format(i), fontsize=font_size2, ha='center', va='baseline')
    plt.text(y_labels_center, i, r'${}$'.format(i), fontsize=font_size2, ha='center', va='center')
    # yticks
    plt.plot([0, tl], [i, i], 'k-')
# zero
plt.text(y_labels_center, x_labels_baseline, r'$0$', fontsize=font_size2, ha='center', va='baseline')


plt.text(1.2, 6.5, r'$\sum_{n=0}^{\infty}\;\sum_{k=0}^{n}$', fontsize=font_size1, ha='left', va='baseline',
         color=(0, .5, 0))
plt.text(1.2, 4.5, r'$\sum_{k=0}^{\infty}\;\sum_{n=k}^{\infty}$', fontsize=font_size1, ha='left', va='baseline',
         color=(0.5, 0, 0))


#p_fancy = FancyBboxPatch((4, 0), 0, 4, boxstyle="round,pad=0.2", fc="none", ec=(0., .5, 0.), zorder=4)
#ax.add_patch(p_fancy)

plt.axis('off')
plt.savefig('montmort_matching_double_summation.pdf', bbox_inches='tight')
plt.show()
