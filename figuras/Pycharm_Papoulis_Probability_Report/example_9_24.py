import matplotlib.pyplot as plt
import numpy as np

__author__ = 'ernesto'

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preview'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

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

# PARAMETROS

omega0 = 6
k = 5.5
T = 1.2
A = 3
N = 400

xmin = -0.5
xmax = 10
ymin = 0
ymax = 6

# construcción del espectro: ventana de hanning
x = np.linspace(xmin, xmax, N)
ns = np.argmax(x > (omega0 - T))
ne = np.argmax(x > (omega0 + T))
S = np.zeros((N, ))
S[ns: ne] = A * (1 + np.cos(np.pi * (x[ns: ne] - omega0) / T)) / 2

# largo de los ejes
xmin_ax = xmin - 0.3
xmax_ax = xmax + 0.6
dy = 0.8
ymin_ax = ymin - dy
ymax_ax = ymax + dy

# parámertos de las figuras
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.8
ytm = 0.2
# font size
fontsize = 13

fig = plt.figure(0, figsize=(10, 5), frameon=False)

ax = plt.subplot2grid((2, 4), (0, 2), rowspan=1, colspan=2)
plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))


plt.annotate("", xytext=(omega0, 0), xycoords='data', xy=(omega0, k), textcoords='data',
             arrowprops=dict(arrowstyle='->, head_length=1, head_width=0.3', lw=2, shrinkA=0, shrinkB=0))
plt.plot([xmin, xmax], [0, 0], color='k', lw=2)

plt.text(xmax_ax, xtm, r'$\omega$', fontsize=fontsize, ha='center', va='baseline')
plt.text(omega0, xtm, r'$\omega_0$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-ytm, xtm, r'$0$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, ymax_ax, r'$S(\omega)$', fontsize=fontsize, ha='left', va='center')
plt.text(6.5, ymax_ax, r'$\textrm{Espectro emitido}$', fontsize=fontsize, ha='left', va='center')

plt.axis('off')


ax = plt.subplot2grid((2, 4), (1, 2), rowspan=1, colspan=2)
plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
htl, vtl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(x, S, color='k', lw=2)

plt.text(xmax_ax, xtm, r'$\omega$', fontsize=fontsize, ha='center', va='baseline')
plt.text(omega0, xtm, r'$\omega_0$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-ytm, xtm, r'$0$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, ymax_ax, r'$S(\omega)$', fontsize=fontsize, ha='left', va='center')
plt.text(6.5, ymax_ax, r'$\textrm{Espectro recibido}$', fontsize=fontsize, ha='left', va='center')
plt.plot([omega0, omega0], [0, vtl], 'k')

plt.axis('off')

max_ax = 10
dx = 1
xO = -14
xP = 0
lv = 5
xm = 10
alpha_gr = 30
alpha = alpha_gr * np.pi / 180
alphas = np.linspace(np.pi - alpha, np.pi + alpha, 100)
ym = xm * np.tan(alpha)
narr = 3
xtm = -1.3
ax = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)
plt.axis('equal')
plt.xlim(xO - dx, xP + lv + dx)

plt.plot([xO, xP], [0, 0], 'k-', lw=1)
plt.plot(xO, 0, 'k.', markersize=14)
plt.plot(xP, 0, 'k.', markersize=14)
plt.annotate("", xytext=(xP, 0), xycoords='data', xy=(xP + lv, 0), textcoords='data',
             arrowprops=dict(width=2, headwidth=6, headlength=12, facecolor='black', shrink=0))

dxi = xm / narr
head_width = 0.36
head_length = 0.6
for i in np.arange(narr):
    xarr = dxi * (i + 0.5)
    yarr = xarr * np.tan(alpha)
    plt.arrow(0, 0, -xarr, yarr, hold=None, width=0.01, length_includes_head=True,
              head_width=head_width, head_length=head_length, head_starts_at_zero=False, overhang=0.2, color='k')
    plt.arrow(0, 0, -xarr, -yarr, hold=None, width=0.01, length_includes_head=True,
              head_width=head_width, head_length=head_length, head_starts_at_zero=False, overhang=0.2, color='k')
    plt.arrow(0, 0, -np.sqrt(xarr ** 2 + yarr ** 2), 0, hold=None, width=0.01, length_includes_head=True,
              head_width=head_width, head_length=head_length, head_starts_at_zero=False, overhang=0.2, color='k')
    xc = dxi * (i + 1)
    radio = np.sqrt((xc ** 2) * (1 + np.tan(alpha) ** 2))
    plt.plot(radio * np.cos(alphas), radio * np.sin(alphas), 'k-', lw=1)
i = narr
xarr = dxi * i
yarr = xarr * np.tan(alpha)
plt.arrow(0, 0, -xarr, yarr, hold=None, width=0.01, length_includes_head=True,
          head_width=0, head_length=0, head_starts_at_zero=False, overhang=0.2, color='k')
plt.arrow(0, 0, -xarr, -yarr, hold=None, width=0.01, length_includes_head=True,
          head_width=0, head_length=0, head_starts_at_zero=False, overhang=0.2, color='k')
plt.arrow(0, 0, -np.sqrt(xarr ** 2 + yarr ** 2), 0, hold=None, width=0.01, length_includes_head=True,
          head_width=0, head_length=0, head_starts_at_zero=False, overhang=0.2, color='k')

# labels
plt.text(xO, xtm, r'$O$', fontsize=fontsize, ha='center', va='baseline')
plt.text(xP, xtm, r'$P$', fontsize=fontsize, ha='center', va='baseline')
plt.text(lv / 2, 0.5, r'$\mathbf{v}$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-3.5, -4, r'$(OP)=\mathbf{r}=r_0+\mathbf{v}t$', fontsize=fontsize, ha='left', va='baseline')
plt.axis('off')

# save as pdf image
plt.savefig('example_9_24.pdf', bbox_inches='tight')
plt.show()