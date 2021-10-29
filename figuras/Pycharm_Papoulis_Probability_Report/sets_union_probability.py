from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)
rc('mathtext', fontset='cm')

colorA = '#0343df'
colorB = '#ff000d'
alpha = 0.4
fontsize = 18

fig = plt.figure(0, figsize=(4, 3), frameon=False)
ax = fig.add_subplot(111)
plt.axis([-1.8, 1.8, -1.3, 1.1])
ax.set_aspect('equal')

ax.add_patch(Circle((-0.5, 0), 1, facecolor=colorA, alpha=0.4, edgecolor=colorA))
ax.add_patch(Circle((0.5, 0), 1, facecolor=colorB, alpha=0.4, edgecolor=colorB))

# labels
plt.text(-1, -1.2, r'$A$', fontsize=fontsize, ha='center', va='baseline')
plt.text(1, -1.2, r'$B$', fontsize=fontsize, ha='center', va='baseline')

plt.text(-1, 0, r'$A\cap B^c$', fontsize=fontsize, ha='center', va='center')
plt.text(1, 0, r'$B\cap A^c$', fontsize=fontsize, ha='center', va='center')
plt.text(0, 0, r'$A\cap B$', fontsize=fontsize, ha='center', va='center')


plt.axis('off')
plt.savefig('sets_union_probability.pdf', bbox_inches='tight')
plt.show()



