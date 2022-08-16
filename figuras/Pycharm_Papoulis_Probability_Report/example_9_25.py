import matplotlib.pyplot as plt
import numpy as np

__author__ = 'ernesto'

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preview'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


# auxiliar function for plot ticks of equal length in x and y axis despite its scales.
# must be invoked after set the axes limits for example with xlim, ylim
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

# Parámetros

T = 0.5
h = 1 / (2 * T)

# parámetros de los ejes
NT = 3
tmin = -NT * T
tmax = NT * T
zmax = 2
zmin = -0.5

delta_t = 0.2
tmin_ax = tmin - delta_t
tmax_ax = tmax + delta_t
zmin_ax = zmin
zmax_ax = zmax


NP = 6.5
wmin = -NP * np.pi / T
wmax = NP * np.pi / T

delta_w = 4
wmin_ax = -NP * np.pi / T - delta_w
wmax_ax = NP * np.pi / T + delta_w
hmax = 1.3
hmin = -0.3
hmax_ax = hmax
hmin_ax = hmin


nw = 800
omega = np.linspace(wmin, wmax, nw)
H = np.sin(omega * T) / (omega * T)


# parámetros de la figura

lw = 2
fontsize = 14

# length of the ticks for all subplot (5 pixels)
display_length = 6  # in pixels

fig = plt.figure(0, figsize=(10, 4.5), frameon=False)
# PULSO RECTANGULAR
# axis labels parameters
bl = -0.5  # x labels baseline
rm = -0.05  # y labels right margin
ax = plt.subplot2grid((3, 10), (0, 0), rowspan=1, colspan=5)
plt.axis([tmin_ax, tmax_ax, zmin_ax, zmax_ax])

# horizontal and vertical ticks length
htl, vtl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(tmin_ax, 0), xycoords='data', xy=(tmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, zmin_ax), xycoords='data', xy=(0, zmax_ax), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# pulso rectangular
plt.plot([tmin, -T], [0, 0], 'k', lw=lw)
plt.plot([-T, -T], [0, h], 'k', lw=lw)
plt.plot([-T, T], [h, h], 'k', lw=lw)
plt.plot([T, T], [0, h], 'k', lw=lw)
plt.plot([T, tmax], [0, 0], 'k', lw=lw)

# axis label
ax.text(tmax_ax, bl, r'$t$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(rm, bl, r'$0$', fontsize=fontsize, ha='right', va='baseline', color='k')
ax.text(0.1, zmax_ax, r'$h(t)$', fontsize=fontsize, ha='left', va='center', color='k')
# pulses labels
ax.text(-T, bl, r'$-T$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(T, bl, r'$T$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(rm, bl, r'$0$', fontsize=fontsize, ha='right', va='baseline', color='k')
ax.text(rm, 1.55, r'$\dfrac{1}{2T}$', fontsize=fontsize, ha='right', va='center', color='k')
plt.axis('off')

# PULSO TRIANGULAR
ax = plt.subplot2grid((3, 10), (0, 5), rowspan=1, colspan=5)
plt.axis([tmin_ax, tmax_ax, zmin_ax, zmax_ax])

# axis arrows
plt.annotate("", xytext=(tmin_ax, 0), xycoords='data', xy=(tmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, zmin_ax), xycoords='data', xy=(0, zmax_ax), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# pulso rectangular
plt.plot([tmin, -2 * T], [0, 0], 'k', lw=lw)
plt.plot([-2 * T, 0], [0, h], 'k', lw=lw)
plt.plot([0, 2 * T], [h, 0], 'k', lw=lw)
plt.plot([2 * T, tmax], [0, 0], 'k', lw=lw)

plt.plot([2 * T, 2 * T], [0, vtl], 'k', lw=1)
plt.plot([-2 * T, -2 * T], [0, vtl], 'k', lw=1)

# axis label
ax.text(tmax_ax, bl, r'$t$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(rm, bl, r'$0$', fontsize=fontsize, ha='right', va='baseline', color='k')
ax.text(0.1, zmax_ax, r'$\rho(t)$', fontsize=fontsize, ha='left', va='center', color='k')
# pulses labels
ax.text(-2*T, bl, r'$-2T$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(2*T, bl, r'$2T$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(rm, bl, r'$0$', fontsize=fontsize, ha='right', va='baseline', color='k')
ax.text(rm, 1.55, r'$\dfrac{1}{2T}$', fontsize=fontsize, ha='right', va='center', color='k')
plt.axis('off')

# axis labels parameters
bl = -0.13  # x labels baseline
rm = -1.5  # y labels right margin

ax = plt.subplot2grid((3, 10), (1, 0), rowspan=2, colspan=5)
plt.axis([wmin_ax, wmax_ax, hmin_ax, hmax_ax])

# horizontal and vertical ticks length
htl, vtl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(wmin_ax, 0), xycoords='data', xy=(wmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, hmin_ax), xycoords='data', xy=(0, hmax_ax), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(omega, H, 'k', lw=lw)

ax.text(wmax_ax, bl, r'$\omega$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(2.2, hmax_ax, r'$H(\omega)$', fontsize=fontsize, ha='left', va='center', color='k')

plt.plot([0, htl], [1, 1], 'k', lw=1)
ax.text(rm, 1.02, r'$1$', fontsize=fontsize, ha='right', va='center', color='k')


plt.annotate(r'$\dfrac{\pi}{T}$', xytext=(2.5 * np.pi / T,  0.4), xycoords='data', xy=(np.pi / T, 0),
             textcoords='data', fontsize=fontsize, va="baseline", ha="left",
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.4", facecolor='black', relpos=(0, 0.2),
                             patchA=None, patchB=None, shrinkA=4, shrinkB=0))
plt.annotate(r'$-\dfrac{\pi}{T}$', xytext=(-2.5 * np.pi / T,  0.4), xycoords='data', xy=(-np.pi / T, 0),
             textcoords='data', fontsize=fontsize, va="baseline", ha="right",
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.4", facecolor='black', relpos=(1, 0.2),
                             patchA=None, patchB=None, shrinkA=4, shrinkB=0))

plt.axis('off')

# Syy

ax = plt.subplot2grid((3, 10), (1, 5), rowspan=2, colspan=5)
plt.axis([wmin_ax, wmax_ax, hmin_ax, hmax_ax])

# axis arrows
plt.annotate("", xytext=(wmin_ax, 0), xycoords='data', xy=(wmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, hmin_ax), xycoords='data', xy=(0, hmax_ax), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(omega, np.square(H), 'k', lw=lw)

ax.text(wmax_ax, bl, r'$\omega$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(2.2, hmax_ax, r'$|H(\omega)|^2$', fontsize=fontsize, ha='left', va='center', color='k')

plt.plot([0, htl], [1, 1], 'k', lw=1)
ax.text(rm, 1.02, r'$1$', fontsize=fontsize, ha='right', va='center', color='k')

plt.plot([np.pi / T, np.pi / T], [0, vtl], 'k', lw=1)
plt.plot([-np.pi / T, -np.pi / T], [0, vtl], 'k', lw=1)
ax.text(np.pi / T, -0.2, r'$\dfrac{\pi}{T}$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(-np.pi / T, -0.2, r'$-\dfrac{\pi}{T}$', fontsize=fontsize, ha='center', va='baseline', color='k')

plt.axis('off')

plt.savefig('example_9_25_transfer.pdf', bbox_inches='tight')

######################################
######################################
######################################

dh = 0.05
hmin_ax = hmin_ax + dh
hmax_ax = hmax_ax + dh
fig = plt.figure(1, figsize=(5, 3), frameon=False)
ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
plt.axis([wmin_ax, wmax_ax, hmin_ax, hmax_ax])

# horizontal and vertical ticks length
htl, vtl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(wmin_ax, 0), xycoords='data', xy=(wmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, hmin_ax), xycoords='data', xy=(0, hmax_ax), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(omega, 1-H, 'k', lw=lw)

ax.text(wmax_ax, bl, r'$\omega$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(2.2, hmax_ax, r"$H'(\omega)$", fontsize=fontsize, ha='left', va='center', color='k')

plt.plot([0, htl], [1, 1], 'k', lw=1)
ax.text(rm, 1.05, r'$1$', fontsize=fontsize, ha='right', va='center', color='k')

ax.text(np.pi / T, -0.2, r'$\dfrac{\pi}{T}$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(-np.pi / T, -0.2, r'$-\dfrac{\pi}{T}$', fontsize=fontsize, ha='center', va='baseline', color='k')

plt.plot([np.pi / T, np.pi / T], [0, 1], 'k', lw=1, linestyle='--', dashes=(5, 3))
plt.plot([-np.pi / T, -np.pi / T], [0, 1], 'k', lw=1, linestyle='--', dashes=(5, 3))
plt.plot([wmin, wmax], [1, 1], 'k', lw=1, linestyle='--', dashes=(5, 3))

plt.axis('off')
plt.savefig('example_9_25_high_pass.pdf', bbox_inches='tight')


######################################
######################################
######################################


fig = plt.figure(2, figsize=(10, 1.3), frameon=False)
# h(t)
# axis labels parameters
bl = -0.5  # x labels baseline
rm = -0.05  # y labels right margin
ax = plt.subplot2grid((1, 10), (0, 0), rowspan=1, colspan=5)
plt.axis([tmin_ax, tmax_ax, zmin_ax, zmax_ax])

# horizontal and vertical ticks length
htl, vtl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(tmin_ax, 0), xycoords='data', xy=(tmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, zmin_ax), xycoords='data', xy=(0, zmax_ax), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# pulso rectangular
plt.plot([tmin, -T], [0, 0], 'k', lw=lw)
plt.plot([-T, -T], [0, h], 'k', lw=lw)
plt.plot([-T, T], [h, h], 'k', lw=lw)
plt.plot([T, T], [0, h], 'k', lw=lw)
plt.plot([T, tmax], [0, 0], 'k', lw=lw)

# axis label
ax.text(tmax_ax, bl, r'$t$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(rm, bl, r'$0$', fontsize=fontsize, ha='right', va='baseline', color='k')
ax.text(0.1, zmax_ax, r'$h(t)$', fontsize=fontsize, ha='left', va='center', color='k')
# pulses labels
ax.text(-T, bl, r'$-T$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(T, bl, r'$T$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(rm, bl, r'$0$', fontsize=fontsize, ha='right', va='baseline', color='k')
ax.text(rm, 1.55, r'$\dfrac{1}{2T}$', fontsize=fontsize, ha='right', va='center', color='k')
plt.axis('off')

# h(t-alpha)
ax = plt.subplot2grid((1, 10), (0, 5), rowspan=1, colspan=5)
plt.axis([tmin_ax, tmax_ax, zmin_ax, zmax_ax])

# axis arrows
plt.annotate("", xytext=(tmin_ax, 0), xycoords='data', xy=(tmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, zmin_ax), xycoords='data', xy=(0, zmax_ax), textcoords='data',
             arrowprops=dict(width=0, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

t0 = 0.8
plt.plot([tmin, t0 - T], [0, 0], 'k', lw=lw)
plt.plot([t0 - T, t0 - T], [0, h], 'k', lw=lw)
plt.plot([t0 - T, t0 + T], [h, h], 'k', lw=lw)
plt.plot([t0 + T, t0 + T], [0, h], 'k', lw=lw)
plt.plot([t0 + T, tmax], [0, 0], 'k', lw=lw)

plt.plot([t0, t0], [0, vtl], 'k', lw=1)
plt.plot([0, htl], [h, h], 'k', lw=1)


# axis label
ax.text(tmax_ax, bl, r'$\alpha$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(rm, bl, r'$0$', fontsize=fontsize, ha='right', va='baseline', color='k')
ax.text(0.1, zmax_ax, r'$h(t-\alpha)$', fontsize=fontsize, ha='left', va='center', color='k')
# pulses labels
ax.text(t0 - T, bl, r'$t-T$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(t0 + T, bl, r'$t+T$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(t0, bl, r'$t$', fontsize=fontsize, ha='center', va='baseline', color='k')
ax.text(rm, bl, r'$0$', fontsize=fontsize, ha='right', va='baseline', color='k')
ax.text(rm, h, r'$\dfrac{1}{2T}$', fontsize=fontsize, ha='right', va='center', color='k')
plt.axis('off')

plt.savefig('example_9_25_rectangular_pulses.pdf', bbox_inches='tight')

plt.show()

