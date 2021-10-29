from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

dim = 10

X, Y = np.meshgrid([-dim, dim], [-dim, dim])
Z = np.zeros((2, 2))

angle = .5
X2, Y2 = np.meshgrid([-dim, dim], [0, dim])
Z2 = Y2 * angle
X3, Y3 = np.meshgrid([-dim, dim], [-dim, 0])
Z3 = Y3 * angle

r = 7
M = 1000
th = np.linspace(0, 2 * np.pi, M)

x, y, z = r * np.cos(th),  r * np.sin(th), angle * r * np.sin(th)

print(X2)
print(Y3)
print(Z3)

ax.plot_surface(X2, Y3, Z3, color='blue', alpha=.5, linewidth=0, zorder=-1)

ax.plot(x[y < 0], y[y < 0], z[y < 0], lw=5, linestyle='--', color='green',
        zorder=0)

ax.plot_surface(X, Y, Z, color='red', alpha=.5, linewidth=0, zorder=1)

ax.plot(r * np.sin(th), r * np.cos(th), np.zeros(M), lw=5, linestyle='--',
        color='k', zorder=2)

ax.plot_surface(X2, Y2, Z2, color='blue', alpha=.5, linewidth=0, zorder=3)

ax.plot(x[y > 0], y[y > 0], z[y > 0], lw=5, linestyle='--', color='green',
        zorder=4)

#plt.axis('off')
plt.show()
