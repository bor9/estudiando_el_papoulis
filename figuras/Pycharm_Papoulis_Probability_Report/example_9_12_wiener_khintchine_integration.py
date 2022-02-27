import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

__author__ = 'ernesto'

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preview'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


def curly_brace(xmin, xmax, y, amp, position='up', beta=8, step=0.01):
    xm = (xmin+xmax)/2
    x_left = np.arange(xmin, xm, step)
    hb_left = amp*(1/(1.+np.exp(-1*beta*(x_left-xmin)))+1/(1.+np.exp(-1*beta*(x_left-xm)))-1/2)
    x = np.concatenate((x_left, np.arange(xm, xmax-step, step)))
    hb = np.concatenate((hb_left, hb_left[-2::-1]))
    if position == 'up':
        return x, hb+y
    elif position == 'down':
        return x, -hb+y
    elif position == 'left':
        return hb, x
    elif position == 'right':
        return -hb, x


T = 5
t_max = 2 * T + 2
tau1 = T + 1.5
tau2 = -T + 2
dtau = 2 * T / 15

# ticks length
tl = t_max/40
# y tick margin
ytm = 0.5
# font size
font_size1 = 18
font_size2 = 12

fig = plt.figure(1, figsize=(8, 8), frameon=False)
ax = fig.add_subplot(111)
max_ax = t_max+0.1
plt.ylim(-max_ax, max_ax)
plt.xlim(-max_ax, max_ax)

# axis arrows
plt.annotate("", xytext=(-t_max, 0), xycoords='data', xy=(t_max, 0), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, -t_max), xycoords='data', xy=(0, t_max), textcoords='data',
             arrowprops=dict(width=0.2, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# axis labels
plt.text(t_max, -0.8, r'$t_1$', fontsize=font_size2, ha='right', va='baseline')
plt.text(ytm, t_max, r'$t_2$', fontsize=font_size2, ha='left', va='top')

# rectangle
plt.plot([-T, T], [T, T], 'k', lw=2)
plt.plot([-T, T], [-T, -T], 'k', lw=2)
plt.plot([-T, -T], [-T, T], 'k', lw=2)
plt.plot([T, T], [-T, T], 'k', lw=2)

plt.text(T+0.2, 0.3, r'$T$', fontsize=font_size2, ha='left', va='baseline')
plt.text(-T-0.2, 0.3, r'$-T$', fontsize=font_size2, ha='right', va='baseline')
plt.text(-0.2, T+0.3, r'$T$', fontsize=font_size2, ha='right', va='baseline')
plt.text(-0.2, -T-0.3, r'$-T$', fontsize=font_size2, ha='right', va='top')

##################
##################
# positive tau
##################
##################
tau1 = T + 1.5
# trapecio - coordenadas de los vertices
a1x = tau1 - T
a1y = -T
b1x = tau1 + dtau - T
b1y = -T
c1x = T
c1y = -(tau1 + dtau - T)
d1x = T
d1y = -(tau1 - T)
plt.plot([a1x, d1x], [a1y, d1y], 'k-')
plt.plot([b1x, c1x], [b1y, c1y], 'k-')
# vertices
vertices2 = np.array([[a1x, a1y], [b1x, b1y], [c1x, c1y], [d1x, d1y]])
ax.add_patch(Polygon(vertices2, color='#CCCCCC'))
# lineas punteadas
t1_end = t_max - 0.2
plt.plot([t1_end, a1x], [t1_end - tau1, a1y], 'k--', dashes=(5, 3), lw=1)
plt.plot([t1_end, b1x], [t1_end - (tau1 + dtau), b1y ], 'k--', dashes=(5, 3), lw=1)
# etiquetas de lineas punteadas
plt.text(t1_end, t1_end - tau1 + 0.7, r'$t_1-t_2=\tau>0$', fontsize=font_size2, rotation=45, ha='right', va='top')
plt.text(t1_end, t1_end - dtau - tau1 - 0.8, r'$t_1-t_2=\tau+\Delta\tau$', fontsize=font_size2, rotation=45, ha='right',
         va='top')

# etiquetas
# corchete rotado
x3, b3 = curly_brace(0, np.sqrt(2)*(2*T-tau1), 0, 0.5, position='up', beta=20, step=0.02)
# plt.plot(x3, b3, 'k')
x3_rot = np.sqrt(np.square(x3) + np.square(b3))*np.sin(np.pi/4 - np.arctan(b3/x3))
b3_rot = np.sqrt(np.square(x3) + np.square(b3))*np.cos(np.pi/4 - np.arctan(b3/x3))
x3_rot += a1x - 0.2
b3_rot += a1y + 0.2
plt.plot(x3_rot, b3_rot, 'k')
plt.text(T/2.2, -T/2.2, r"$\sqrt{2}(2T-\tau)$", fontsize=font_size2, ha='center', va='center', rotation=45)
# altura del trapecio
plt.plot([c1x, c1x-dtau/2], [c1y, c1y+dtau/2], 'k')
ax.annotate(r"$\Delta\tau/\sqrt{2}$", xy=((2*c1x-dtau/2)/2-0.05, (2*c1y+dtau/2)/2+0.05), xycoords='data',
            xytext=(T + 4, -5), textcoords='data',
            va="center", ha="right", fontsize=font_size2,
            arrowprops=dict(arrowstyle="->", color="k", shrinkA=25, shrinkB=1, patchA=None,
                            patchB=None, connectionstyle="angle3,angleA=0,angleB=70"))
# corchete
x_br1, y_br1 = curly_brace(a1x, d1x, -T - 0.2, 0.5, position='down', beta=20, step=0.02)
plt.plot(x_br1, y_br1, 'k')
plt.text((a1x + d1x) / 2 + T/2, -T - 1, r"$T-\left(\tau-T\right)=2T-\tau$", fontsize=font_size2,
         ha='center', va='top')
# corchete tau - T
x1, b1 = curly_brace(0, tau1 - T, -T - 1, 0.5, position='down', beta=20, step=0.01)
plt.plot(x1, b1, 'k')
plt.text((tau1 - T)/2 + 0.2, -T - 1.8, r"$\tau-T$", fontsize=font_size2, ha='center', va='top')

# xticks
plt.plot([tau1, tau1], [0, tl], 'k-')
plt.plot([tau1+dtau, tau1+dtau], [0, tl], 'k-')
# xlabels
plt.text(tau1, 0.5, r'$\tau$', fontsize=font_size2, ha='right', va='baseline')
plt.text(tau1+dtau, -0.8, r'$\tau+\Delta\tau$', fontsize=font_size2, ha='left', va='baseline')

##################
##################
# negative tau
##################
##################
tau2 = -T/2 - 1
# trapecio - coordenadas de los vertices
a2x = -T
a2y = -(tau2 + T)
b2x = -T
b2y = -(tau2 + dtau + T)
c2x = T - (-tau2 - dtau)
c2y = T
d2x = T + tau2
d2y = T
plt.plot([a2x, d2x], [a2y, d2y], 'k-')
plt.plot([b2x, c2x], [b2y, c2y], 'k-')
# vertices
vertices2 = np.array([[a2x, a2y], [b2x, b2y], [c2x, c2y], [d2x, d2y]])
ax.add_patch(Polygon(vertices2, color='#CCCCCC'))
# lineas punteadas
t2_end = 1.5 * T
plt.plot([t2_end, a2x], [t2_end - tau2, a2y], 'k--', lw=1, dashes=(5, 3))
plt.plot([t2_end, b2x], [t2_end - (tau2 + dtau), b2y ], 'k--', lw=1, dashes=(5, 3))
# etiquetas de lineas punteadas
plt.text(t2_end, t2_end - tau2 + 0.7, r"$t_1-t_2=\tau'<0$", fontsize=font_size2, rotation=45, ha='right', va='top')
plt.text(t2_end, t2_end - dtau - tau2 - 0.8, r"$t_1-t_2=\tau'+\Delta\tau'$", fontsize=font_size2, rotation=45,
         ha='right', va='top')

# xticks
plt.plot([tau2, tau2], [0, tl], 'k-')
plt.plot([tau2+dtau, tau2+dtau], [0, tl], 'k-')
# xlabels
plt.text(tau2, 0.5, r"$\tau'$", fontsize=font_size2, ha='right', va='baseline')
plt.text(tau2+dtau, -0.8, r"$\tau'+\Delta\tau'$", fontsize=font_size2, ha='left', va='baseline')

plt.axis('off')
# save as pdf image
plt.savefig('example_9_12_wiener_khintchine_integration.pdf', bbox_inches='tight')
plt.show()
