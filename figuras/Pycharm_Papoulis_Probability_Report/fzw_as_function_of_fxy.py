import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

__author__ = 'ernesto'

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preview'] = True


###############
# PARAMETEROS #
###############

# Límites de los ejes
xmin_ax = -2
xmax_ax = 6
ymin_ax = xmin_ax
ymax_ax = xmax_ax

# Rectángulo en el plano zw
z = 2
w = 2
dz = 1.5
dw = 1.5

# Paralelogramos en el plano xy
# base
b = 1.5
h = 1
a = 0.5

# coordenadas del vértice inferior izquierdo de los paralelogramos
xis = np.array([0.9, 1.6, 2.5, 3.2])
yis = np.array([2, -1.3, 4.5, 1])

# transformación lineal entre
T = np.array([3, 1, 2, 2])
T = T * 0.6


######################
# FIN DE PARAMETEROS #
######################


# coordenadas de los vértices del cuadrado en el plano zw
zs = np.array([z, z, z + dz, z + dz])
ws = np.array([w, w + dw, w + dw, w])

# coordenadas de los vértices del paralelogramo en el plano xy
xs = T[0] * zs + T[1] * ws - 4
ys = T[2] * zs + T[3] * ws - 3.5


##############################
# PARÁMETROS DE LAS GRAFICAS #
##############################

# ancho de línea
lw = 2
fontsize = 12
# baseline
xbl = -0.5
yrm = -0.3


##############
#    PLOT    #
##############

#
# Figura de mapeo a varios puntos
#
fig = plt.figure(0, figsize=(7, 4), frameon=False)

## PLANO zw
ax = plt.subplot2grid((1, 6), (0, 0), rowspan=1, colspan=3)

plt.axis([xmin_ax, xmax_ax, ymin_ax, ymax_ax])
ax.set_aspect('equal', adjustable='box')

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=5, headlength=7, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=5, headlength=7, facecolor='black', shrink=0.002))

# dibujo del cuadrado y vértice
ax.add_patch(patches.Polygon(xy=list(zip(zs, ws)), fill=False, lw=lw))
plt.plot(z, w, 'k.', ms=8)
# etiquetas
plt.text(z, w - 0.15, '$(z,\,w)$', fontsize=fontsize, ha='right', va='top')
plt.text(z + dz, w - 0.15, '$(z+\Delta z,\,w)$', fontsize=fontsize, ha='left', va='top')
plt.text(z + 0.7, w + dw + 0.15, '$(z,\,w+\Delta w)$', fontsize=fontsize, ha='right', va='bottom')

plt.text(z + 0.15, w + 0.15, '$A$', fontsize=fontsize, ha='left', va='baseline')
plt.text(z - 0.15 + dz, w + 0.15, '$B$', fontsize=fontsize, ha='right', va='baseline')
plt.text(z + 0.15, w + dw - 0.5, '$C$', fontsize=fontsize, ha='left', va='baseline')
plt.text(z - 0.15 + dz, w + dw - 0.5, '$D$', fontsize=fontsize, ha='right', va='baseline')

plt.text(xmax_ax, xbl, '$z$', fontsize=fontsize, ha='center', va='baseline')
plt.text(yrm, ymax_ax, '$w$', fontsize=fontsize, ha='right', va='center')

plt.axis('off')

## PLANO xy
ax = plt.subplot2grid((1, 6), (0, 3), rowspan=1, colspan=3)

plt.axis([xmin_ax, xmax_ax, ymin_ax, ymax_ax])
ax.set_aspect('equal', adjustable='box')

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=5, headlength=7, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=5, headlength=7, facecolor='black', shrink=0.002))

# arreglo con los subindices como strings
si = np.arange(1, xis.shape[0] - 1).astype(str)
si = np.append(si, ['i', 'n'])

for i, xi in np.ndenumerate(xis):
    yi = yis[i]
    # coordenadas de los vértices del poligono
    x = [xi, xi + a, xi + a + b, xi + b]
    y = [yi, yi + h, yi + h, yi]
    # dibujo del poligono y vértice
    ax.add_patch(patches.Polygon(xy=list(zip(x, y)), fill=False, lw=lw))
    plt.plot(xi, yi, 'k.', ms=8)
    # etiquetas
    plt.text(xi + (a + b) / 2, yi + h / 2, '$\Delta_{}$'.format(si[i]), fontsize=fontsize, ha='center', va='center')
    plt.text(xi, yi-0.15, '$(x_{0:},\,y_{0:})$'.format(si[i]), fontsize=fontsize, ha='center', va='top')

plt.text(xmax_ax, xbl, '$x$', fontsize=fontsize, ha='center', va='baseline')
plt.text(yrm, ymax_ax, '$y$', fontsize=fontsize, ha='right', va='center')

plt.axis('off')

# save as pdf image
plt.savefig('fzw_as_function_of_fxy_1.pdf', bbox_inches='tight')


#
# Figura para el cálculo del mapeo del área
#
fig = plt.figure(1, figsize=(7, 4), frameon=False)


## PLANO zw
ax = plt.subplot2grid((1, 6), (0, 0), rowspan=1, colspan=3)

plt.axis([xmin_ax, xmax_ax, ymin_ax, ymax_ax])
ax.set_aspect('equal', adjustable='box')

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=5, headlength=7, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=5, headlength=7, facecolor='black', shrink=0.002))

# dibujo del cuadrado - cambio de tamaño y posición
z = 2
w = 3
dz = 2
dw = 2
zs = np.array([z, z, z + dz, z + dz])
ws = np.array([w, w + dw, w + dw, w])
ax.add_patch(patches.Polygon(xy=list(zip(zs, ws)), fill=False, lw=lw))

plt.plot([z, z], [0, w], 'k--', lw=1)
plt.plot([z + dz, z + dz], [0, w], 'k--', lw=1)
plt.plot([0, z], [w, w], 'k--', lw=1)
plt.plot([0, z], [w + dw, w + dw], 'k--', lw=1)

# etiquetas
plt.text(z - 0.1, w - 0.2, '$A$', fontsize=fontsize, ha='right', va='top')
plt.text(z + dz + 0.1, w - 0.2, '$B$', fontsize=fontsize, ha='left', va='top')
plt.text(z, w + dw + 0.05, '$C$', fontsize=fontsize, ha='right', va='bottom')
plt.text(z + dz + 0.1, w + dw + 0.05, '$D$', fontsize=fontsize, ha='left', va='bottom')

plt.text((2 * z + dz) / 2, w / 2, '$\Delta z$', fontsize=fontsize, ha='center', va='center')
plt.text(z / 2, (2 * w + dw) / 2, '$\Delta w$', fontsize=fontsize, ha='center', va='center')

plt.text(z, xbl, '$z$', fontsize=fontsize, ha='center', va='baseline')
plt.text(z + dz, xbl, '$z+\Delta z$', fontsize=fontsize, ha='center', va='baseline')
plt.text(yrm, w, '$w$', fontsize=fontsize, ha='right', va='center')
plt.text(yrm, w + dw, '$w+\Delta w$', fontsize=fontsize, ha='right', va='center')

plt.text(xmax_ax, xbl, '$z$', fontsize=fontsize, ha='center', va='baseline')
plt.text(yrm, ymax_ax, '$w$', fontsize=fontsize, ha='right', va='center')

plt.axis('off')


## PLANO xy
ax = plt.subplot2grid((1, 6), (0, 3), rowspan=1, colspan=3)

plt.axis([xmin_ax, xmax_ax, ymin_ax, ymax_ax])
ax.set_aspect('equal', adjustable='box')

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=5, headlength=7, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=5, headlength=7, facecolor='black', shrink=0.002))

# dibujo del paralelogramo
ax.add_patch(patches.Polygon(xy=list(zip(xs, ys)), fill=False, lw=lw))

plt.plot([xs[0], xs[0]], [0, ys[0]], 'k--', lw=1)
plt.plot([0, xs[0]], [ys[0], ys[0]], 'k--', lw=1)

# ángulos
plt.plot([xs[0], 3], [ys[0], ys[0]], 'k-', lw=1)
phi = np.arctan((ys[1] - ys[0]) / (xs[1] - xs[0]))
r = 1
phis = np.linspace(0, phi, 50)
plt.plot(xs[0] + r * np.cos(phis), ys[0] + r * np.sin(phis), 'k-', lw=1)
phi = np.arctan((ys[3] - ys[0]) / (xs[3] - xs[0]))
r = 1.5
phis = np.linspace(0, phi, 50)
plt.plot(xs[0] + r * np.cos(phis), ys[0] + r * np.sin(phis), 'k-', lw=1)

# etiquetas
plt.text(xs[0] - 0.1, ys[0] - 0.2, "$A'$", fontsize=fontsize, ha='right', va='top')
plt.text(xs[1], ys[1]+ 0.05, "$C'$", fontsize=fontsize, ha='right', va='bottom')
plt.text(xs[2] + 0.1, ys[2] + 0.05, "$D'$", fontsize=fontsize, ha='left', va='bottom')
plt.text(xs[3] + 0.1, ys[3] - 0.2, "$B'$", fontsize=fontsize, ha='left', va='top')

# ángulos
plt.text(1.65, 2.25, "$\\theta$", fontsize=fontsize, ha='center', va='center')
plt.text(2.5, 1.8, "$\phi$", fontsize=fontsize, ha='center', va='center')

# área
plt.text((xs[0] + xs[2]) / 2, (ys[0] + ys[2]) / 2, "$\Delta_i$", fontsize=fontsize, ha='center', va='center')

# etiquetas de los ejes
plt.text(xs[0], xbl, '$x_i$', fontsize=fontsize, ha='center', va='baseline')
plt.text(yrm, ys[0], '$y_i$', fontsize=fontsize, ha='right', va='center')

plt.text(xmax_ax, xbl, '$x$', fontsize=fontsize, ha='center', va='baseline')
plt.text(yrm, ymax_ax, '$y$', fontsize=fontsize, ha='right', va='center')

plt.axis('off')

plt.savefig('fzw_as_function_of_fxy_2.pdf', bbox_inches='tight')

plt.show()
