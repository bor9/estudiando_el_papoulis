import numpy as np

__author__ = 'ernesto'

m = 40
n = 100
k = 5

# P(W_k) computation

ks = np.arange(k-1)
pw = 0
tmp = 1
for ki in ks:
    tmp *= (n - ki) / (m + n - (ki + 1))
    pw += tmp
pw = m / (m + n) * (1 + pw)

# P(Y_k) computation
ks = np.arange(k)
py = 1
for ki in ks:
    py *= (n - ki) / (m + n - ki)
py = 1 - py

print(pw-py)