import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib import cm

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


# auxiliar function for plot ticks of equal length in x and y axis despite its scales.
def convert_display_to_data_coordinates(transData, length=10):
    # create a transform which will take from display to data coordinates
    inv = transData.inverted()
    # transform from display coordinates to data coordinates in x axis
    data_coords = inv.transform([(0, 0), (length, 0)])
    # get the length of the segment in data units
    x_coord_len = data_coords[1, 0] - data_coords[0, 0]
    # transform from display coordinates to data coordinates in y axis
    data_coords = inv.transform([(0, 0), (0, length)])
    # get the length of the segment in data units
    y_coord_len = data_coords[1, 1] - data_coords[0, 1]
    return x_coord_len, y_coord_len

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

# número de muestras de los ejes \bar{x} y \lambda
ns = 400
x = np.linspace(xmin, xmax, ns)
lam = np.linspace(lmin, lmax, ns)

# grilla
xx, ll = np.meshgrid(x, lam)
# forma cuadrática: (\lambda - \bar{x})^2. no se usa.
quad1 = (ll - xx) ** 2

delta = 1 - gamma
z_u = norm.ppf(1 - delta / 2, loc=0, scale=1)
A = z_u ** 2 / n
# plano: A\lambda
plane = A * ll

# rotación de la grilla para mejor visualización de la superficie cuadrática
phi = np.pi/4
xm = xx * np.cos(phi) + ll * np.sin(phi)
ym = -xx * np.sin(phi) + ll * np.cos(phi)
# forma cuadrática: (\lambda - \bar{x})^2 en la grilla rotada.
quad2 = (ym - xm) ** 2

# parábola en eje rotado pi/4
y = np.linspace(-0.8, 0.5, 300)
B = 2 * np.sqrt(2) / A
z = B * (y ** 2) + y
# rotación para obtener la parábola en el eje de interés (\bar{x}, \lambda)
phi = -np.pi/4
xr = y * np.cos(phi) - z * np.sin(phi)
lr = y * np.sin(phi) + z * np.cos(phi)

idx1 = np.argmax(lr < lmax)
xr_trim = xr[idx1:]
lr_trim = lr[idx1:]
idx2 = np.argmax(xr_trim > xmax)
xr_trim = xr_trim[:idx2-1]
lr_trim = lr_trim[:idx2-1]

# parte del plano al frente de la cuadrática
# hay que incrementar la resolución para suavizar los bordes.
ns2 = 1000
# ns2 = 400
x = np.linspace(xmin, xmax, ns2)
lam = np.linspace(lmin, lmax, ns2)

xx2, ll2 = np.meshgrid(x, lam)
plane_front = np.where((ll2 - xx2) ** 2 - A * ll2 < 0, A * ll2, np.nan)

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
alpha = 0.8
fontsize = 13

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

# plano del frente
surf_font = ax.plot_surface(xx2, ll2, plane_front, alpha=alpha, cmap=cm.autumn, linewidth=0, antialiased=True,
                            rstride=10, cstride=10, zorder=10,
                            vmin=np.nanmin(plane), vmax=np.nanmax(plane))

# parábola
plt.plot(xr_trim, lr_trim, A * lr_trim, 'k', lw=1.5)
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
plt.plot(xr_trim, lr_trim, zmin_ax, 'k', lw=1.5)
plt.plot([0, xmax], [0, xmax], [zmin_ax, zmin_ax], 'k--', lw=1)

dax = -0.2
arw = Arrow3D([dax, xmax_ax], [0, 0], [zmin_ax, zmin_ax], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=10)
ax.add_artist(arw)
arw = Arrow3D([0, 0], [dax, lmax_ax], [zmin_ax, zmin_ax], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=10)
ax.add_artist(arw)


ax.view_init(elev=27, azim=21)
# Distance view. Default is 10.
ax.dist = 8

# labels
ax.text(xmax_ax, -0.05, 0, r'$\bar{\mathbf{x}}$', fontsize=fontsize, ha='center', va='center')
ax.text(0.1, lmax_ax + 0.07, 0, r'$\lambda$', fontsize=fontsize, ha='center', va='center')
ax.text(xmax_ax, -0.05, zmin_ax, r'$\bar{\mathbf{x}}$', fontsize=fontsize, ha='center', va='center')
ax.text(0.1, lmax_ax + 0.07, zmin_ax, r'$\lambda$', fontsize=fontsize, ha='center', va='center')

ax.text(xmax + 0.1, lmax + 0.08, 0, r'$\lambda=\bar{\mathbf{x}}$', fontsize=fontsize, ha='left', va='center')
ax.text(xmax + 0.1, lmax + 0.08, zmin_ax, r'$\lambda=\bar{\mathbf{x}}$', fontsize=fontsize, ha='left', va='center')

ax.text(0, lmax, zmax_ax + 0.2, r'$(\lambda-\bar{\mathbf{x}})^2$', fontsize=fontsize, ha='left', va='center')
ax.text(2 * xmax / 3, lmin, 0, r'$\displaystyle\frac{z_u^2}{n}\lambda$', fontsize=fontsize, ha='center', va='center')

plt.axis('off')

##############################################################################

# ramas de la parábola
lr_up = lr[xr >= 0][lr[xr >= 0] > xr[xr >= 0]]
xr_up = xr[xr >= 0][lr[xr >= 0] > xr[xr >= 0]]
lr_down = lr[xr >= 0][lr[xr >= 0] < xr[xr >= 0]]
xr_down = xr[xr >= 0][lr[xr >= 0] < xr[xr >= 0]]

# margen de etiquetas del eje horizontal
hlm = -0.17
# margen de etiquetas del eje vertical
vlm = -0.08
# valor de la media muestral
bar_x = 0.8
# intervalo de confianza
l12 = np.roots([1, -2 * bar_x - A, bar_x ** 2])
print(l12)


ax = plt.subplot2grid((1, 10), (0, 7), rowspan=1, colspan=3)
ax.axis('equal')

# proyección en el plano (x, lambda)
plt.plot(xr[xr < 0], lr[xr < 0], 'k--', lw=2)
plt.plot(xr_up, lr_up, 'k', lw=2)
plt.plot(xr_down, lr_down, 'k', lw=2)


xmin_ax, xmax_ax = ax.get_xlim()
ymin_ax, ymax_ax = ax.get_ylim()

# horizontal and vertical ticks length
display_length = 7
htl, vtl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=5, headlength=6, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=5, headlength=6, facecolor='black', shrink=0.002))

plt.plot([0, xr[-1]], [0, xr[-1]], 'k--', lw=1)

plt.plot([0, htl], [A, A], 'k', lw=1)
plt.plot([bar_x, bar_x], [0, vtl], 'k', lw=1)
plt.plot([bar_x, bar_x], [l12[1], l12[0]], 'r', lw=2)
plt.plot([0, bar_x], [l12[0], l12[0]], 'k--', lw=1)
plt.plot([0, bar_x], [l12[1], l12[1]], 'k--', lw=1)

# labels
plt.text(xmax_ax, hlm, r'$\bar{\mathbf{x}}$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-vlm, ymax_ax, r'$\lambda$', fontsize=fontsize, ha='left', va='center')
plt.text(vlm, A, r'$\frac{z_u^2}{n}$', fontsize=fontsize, ha='right', va='center')
plt.text(bar_x + 0.05, l12[1]-0.07, r'$\lambda_1$', fontsize=fontsize, ha='left', va='center')
plt.text(bar_x + 0.05, l12[0]-0.05, r'$\lambda_2$', fontsize=fontsize, ha='left', va='center')

plt.axis('off')

plt.savefig('poisson_confidence_interval_v3.pdf', bbox_inches='tight')
plt.show()


