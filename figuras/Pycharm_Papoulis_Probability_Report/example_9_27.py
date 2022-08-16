import matplotlib.pyplot as plt
import numpy as np

__author__ = 'ernesto'

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preview'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

# Parámetros

q = 1
b = np.array([0.5, 2, 4])
c = 2 / b
print(b)
print(c)

tmax = 25
wmax = 5

# Fin de parámetros

t = np.linspace(0, tmax, 1000)
w = np.linspace(-wmax, wmax, 1000)

n = b.shape[0]
nt = t.shape[0]
nw = w.shape[0]

Syy = np.zeros((nw, n))
Ryy = np.zeros((nt, n))

for i in np.arange(n):
    Syy[:, i] = q / (np.square(c[i] - np.square(w)) + (b[i] ** 2) * np.square(w))

# b^2 < 4c
i = 0
alpha = b[i] / 2
beta = np.sqrt(4 * c[i] - b[i] ** 2) / 2
Ryy[:, i] = q / (2 * b[i] * c[i]) * np.exp(-alpha * t) * (np.cos(beta * t) + (alpha / beta) * np.sin(beta * t))

# b^2 = 4c
i = 1
alpha = b[i] / 2
Ryy[:, i] = q / (2 * b[i] * c[i]) * np.exp(-alpha * t) * (1 + alpha * t)

# b^2 > 4c
i = 2
alpha = b[i] / 2
gamma = np.sqrt(b[i]**2 - 4 * c[i]) / 2
Ryy[:, i] = q / (4 * b[i] * c[i] * gamma) * ((alpha + gamma) * np.exp(-(alpha - gamma) * t)
                                             - (alpha - gamma) * np.exp(-(alpha + gamma) * t))

lw = 2
fs = 12
N = 3
dw = 0.08
dt = 0.08
tt = 16 # posición del texto
fig = plt.figure(0, figsize=(8, 4), frameon=False)
i = 0
ax = plt.subplot2grid((N * n, N * 2), (i * N, 0), rowspan=N, colspan=N)
plt.axis([-wmax, wmax, 0, np.amax(Syy[:, i]) * (1 + dw)])
plt.plot(w, Syy[:, i], 'k-', lw=lw)
ax.set_xticklabels([])
plt.title(r'$S_{yy}(\omega)$', fontsize=fs)

ax = plt.subplot2grid((N * n, N * 2), (i * N, N), rowspan=N, colspan=N)
plt.axis([0, tmax, np.amin(Ryy[:, i]) * (1 + dt), np.amax(Ryy[:, i]) * (1 + dt)])
plt.plot(t, Ryy[:, i], 'k-', lw=lw)
ax.set_xticklabels([])
ax.yaxis.tick_right()
plt.title(r'$R_{yy}(\tau)$', fontsize=fs)
plt.text(tt, np.amax(Ryy[:, i]) - 0.01, '$b^2<4c$\n$b={:.1f},\;c={:.0f}$'.format(b[i], c[i]), fontsize=fs,
         ha='left', va='top', ma='left')

i = 1
ax = plt.subplot2grid((N * n, N * 2), (i * N, 0), rowspan=N, colspan=N)
plt.axis([-wmax, wmax, 0, np.amax(Syy[:, i]) * (1 + dw)])
plt.plot(w, Syy[:, i], 'k-', lw=lw)

ax = plt.subplot2grid((N * n, N * 2), (i * N, N), rowspan=N, colspan=N)
plt.axis([0, tmax, 0, np.amax(Ryy[:, i]) * (1 + dt)])
plt.plot(t, Ryy[:, i], 'k-', lw=lw)
ax.set_xticklabels([])
ax.yaxis.tick_right()
plt.text(tt, np.amax(Ryy[:, i]), '$b^2=4c$\n$b={:.0f},\;c={:.0f}$'.format(b[i], c[i]), fontsize=fs,
         ha='left', va='top', ma='left')


i = 2
ax = plt.subplot2grid((N * n, N * 2), (i * N, 0), rowspan=N, colspan=N)
plt.axis([-wmax, wmax, 0, np.amax(Syy[:, i]) * (1 + dw)])
plt.plot(w, Syy[:, i], 'k-', lw=lw)
plt.xlabel(r'$\omega$', fontsize=fs)

ax = plt.subplot2grid((N * n, N * 2), (i * N, N), rowspan=N, colspan=N)
plt.axis([0, tmax, 0, np.amax(Ryy[:, i]) * (1 + dt)])
plt.plot(t, Ryy[:, i], 'k-', lw=lw)
ax.yaxis.tick_right()
plt.xlabel(r'$\tau$', fontsize=fs)
plt.text(tt, np.amax(Ryy[:, i]), '$b^2>4c$\n$b={:.0f},\;c={:.1f}$'.format(b[i], c[i]), fontsize=fs,
         ha='left', va='top', ma='left')

plt.savefig('example_9_27.pdf', bbox_inches='tight')

plt.show()


