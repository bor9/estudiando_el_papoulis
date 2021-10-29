import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb
from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)


def win_probability_even(n, p):
    k = np.arange(n + 1, 2*n+1)
    return np.sum(comb(2*n, k)*(p**k)*((1-p)**(2*n-k)))


def win_probability_arbitrary(n, p):
    if n % 2 == 0:
        min_wins = n / 2 + 1
    else:
        min_wins = np.ceil(n/2)
    k = np.arange(min_wins, n+1)
    return np.sum(comb(n, k)*(p**k)*((1-p)**(n-k)))


def get_ticks_positions(mi, ma, step):
    return np.arange(np.ceil(mi / step) * step, ma, step)

nmax = 100
ps = np.array([0.3, 0.47, 0.48, 0.5])
n = np.arange(1, nmax+1)
p2n = np.zeros(shape=(nmax, len(ps)))

for j in np.arange(len(ps)):
    for i in np.arange(nmax):
        p2n[i, j] = win_probability_even(n[i], ps[j])

# optimun n
n_opt = np.zeros(len(ps)-1)
for i in np.arange(len(n_opt)):
    s = 1/(1 - 2*ps[i])
    if s.is_integer():
        if s % 2 == 0:
            n_opt[i] = s/2
        else:
            n_opt[i] = (s - 1)/2
    else:
        s = np.floor(s)
        if s % 2 == 0:
            n_opt[i] = s/2
        else:
            n_opt[i] = (s + 1)/2


fontsize = 14
bggrey = 0.97

fig = plt.figure(1, figsize=(8, 7), frameon=False)
ax = plt.subplot2grid((3, 2), (0, 0), rowspan=2, colspan=2)

plt.plot(n, p2n, linewidth=1.5)
leg = plt.legend(['$p={}$'.format(p) for p in ps], loc='lower right', fontsize=12)
leg.get_frame().set_facecolor(bggrey*np.ones((3,)))
leg.get_frame().set_edgecolor(bggrey*np.ones((3,)))
plt.xlim(n[0], n[-1])
plt.text((n[-1]+n[0])/2, -0.04, r'$n$', fontsize=fontsize, ha='center', va='baseline')
plt.ylabel('$P_{2n}$', fontsize=fontsize)
plt.title('$P_{2n}$ para distintos valores de $p$', fontsize=fontsize)

i = 1
ax = plt.subplot2grid((3, 2), (2, 0), rowspan=1, colspan=1)
plt.plot(n, p2n[:, i], 'gs-', markersize=5, markeredgewidth=0.0)
axis1 = [5, 13, 0.308, 0.3115]
plt.axis(axis1)
plt.yticks(get_ticks_positions(axis1[2], axis1[3], 0.001))
plt.xlabel('$n$', fontsize=fontsize)
plt.ylabel('$P_{2n}$', fontsize=fontsize)
plt.text(0.95, 0.93, '$p={}$'.format(ps[i]), transform=ax.transAxes, fontsize=12, fontweight='bold', va='top',
         ha='right', backgroundcolor=bggrey*np.ones((3,)))
plt.grid()

i = 2
ax = plt.subplot2grid((3, 2), (2, 1), rowspan=1, colspan=1)
plt.plot(n, p2n[:, i], 'rs-', markersize=5, markeredgewidth=0.0)
axis2 = [9, 17, 0.3428, 0.3442]
plt.axis(axis2)
plt.yticks(get_ticks_positions(axis2[2], axis2[3], 0.001))
plt.xlabel('$n$', fontsize=fontsize)
plt.text(0.95, 0.93, '$p={}$'.format(ps[i]), transform=ax.transAxes, fontsize=12, fontweight='bold', va='top',
         ha='right', backgroundcolor=bggrey*np.ones((3,)))
plt.grid()

plt.savefig('unfair_win_probabilities_v2.pdf', bbox_inches='tight')
plt.show()
