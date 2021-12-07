import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Polygon, Rectangle
import matplotlib.colors as colors

from matplotlib import cm
from matplotlib import rc

__author__ = 'ernesto'

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preview'] = True


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

#####################################
# PARAMETROS - Puede ser modificado #
#####################################

# percentil
gamma = 0.9

# número de muestras
n = 5

# rangos de interés
# media muestral
xmin = -1
xmax = 1
# lambda
lmin = -1
lmax = 1

#####################
# FIN DE PARAMETROS #
#####################

ns = 400
x = np.linspace(xmin, xmax, ns)
lam = np.linspace(lmin, lmax, ns)

xx, ll = np.meshgrid(x, lam)
quad1 = (ll - xx) ** 2
delta = 1 - gamma
z_u = norm.ppf(1 - delta / 2, loc=0, scale=1)
plane = (z_u ** 2) * ll / n


phi = np.pi/4
xm = xx * np.cos(phi) + ll * np.sin(phi)
ym = -xx * np.sin(phi) + ll * np.cos(phi)
print(xm)

quad2 = (ym - xm) ** 2

###############

y = np.linspace(-0.6, 0.6, 300)
A = z_u ** 2 / n
B = 2 * np.sqrt(2) / A
z = B * (y ** 2) + y
# rotación
phi = -np.pi/4
xr = y * np.cos(phi) - z * np.sin(phi)
lr = y * np.sin(phi) + z * np.cos(phi)

idx1 = np.argmax(lr < lmax)
idx2 = np.argmax(xr > xmax)
xr = xr[idx1:idx2-1]
lr = lr[idx1:idx2-1]

###############


# Parámetros de la gráfica
delta = 0.5
xmin_ax = xmin
xmax_ax = xmax + delta
lmin_ax = lmin
lmax_ax = lmax + delta
offset = -1.5
zmax = 2
zmin_ax = offset
zmax_ax = zmax + delta
# transparencia
alpha = 0.6
fontsize = 15

fig = plt.figure(0, figsize=(10, 5), frameon=False)
ax = plt.subplot2grid((1, 10), (0, 0), rowspan=1, colspan=7, projection='3d')
# customize the z axis.
ax.set_xlim(xmin_ax, xmax_ax)
ax.set_ylim(lmin_ax, lmax_ax)
ax.set_zlim(zmin_ax, zmax_ax)
# grafica 3D
# superficie cuadrática
surf_quad = ax.plot_surface(xm, ym, quad2, alpha=alpha, cmap=cm.coolwarm, linewidth=0, antialiased=True,
                            rstride=10, cstride=10, zorder=1)

# plano
surf_plane = ax.plot_surface(xx, ll, plane, alpha=alpha, cmap=cm.autumn, linewidth=0, antialiased=True,
                             rstride=10, cstride=10, zorder=10)


# parábola
plt.plot(xr, lr, A * lr, 'k', lw=1.5)
plt.plot([xmin, xmax], [xmin, xmax], [0, 0], 'k--', lw=1)


# axis arrows
dax = -0.2
arw = Arrow3D([dax, xmax_ax], [0, 0], [0, 0], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=10)
ax.add_artist(arw)
arw = Arrow3D([0, 0], [dax, lmax_ax], [0, 0], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=10)
ax.add_artist(arw)
arw = Arrow3D([0, 0], [0, 0], [dax, zmax_ax], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=10)
ax.add_artist(arw)

# proyección en el plano (x, labda)
plt.plot(xr, lr, zmin_ax, 'k', lw=1.5)
plt.plot([0, xmax], [0, xmax], [zmin_ax, zmin_ax], 'k--', lw=1)

dax = -0.2
arw = Arrow3D([dax, xmax_ax], [0, 0], [zmin_ax, zmin_ax], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=10)
ax.add_artist(arw)
arw = Arrow3D([0, 0], [dax, lmax_ax], [zmin_ax, zmin_ax], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=10)
ax.add_artist(arw)

plt.xlabel('x')

ax.view_init(elev=22, azim=-10)
# Distance view. Default is 10.
ax.dist = 8

plt.axis('off')

#plt.savefig('poisson_confidence_interval_v1.pdf', bbox_inches='tight')
plt.show()

