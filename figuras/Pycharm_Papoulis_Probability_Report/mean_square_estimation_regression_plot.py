import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.signal

from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)


def design_signal(num_samples):
    # abscissas and ordinates
    xi = [0, 1, 2, 4, 6, 9, 11, 12]
    yi = [0.6, 0.62, 0.68, 0.7, -0.75, 0.4, -0.5, -0.4]
    # linear interpolation of (xi, yi)
    x = np.linspace(xi[0], xi[-1], num=num_samples)
    y = scipy.interpolate.interp1d(xi, yi, kind='linear')(x)
    # fir filter for smoothing
    # scipy.signal.firwin(numtaps, cutoff, width=None, window='hamming', pass_zero=True, scale=True, nyq=1.0)
    taps = scipy.signal.firwin(int(num_samples/5), 10/num_samples, width=None, window='hamming',
                               pass_zero=True, scale=True, nyq=1.0)
    y = scipy.signal.filtfilt(taps, 1, y)
    y /= np.amax(np.fabs(y))
    return y


num_samples = 1001

# abscissas and ordinates
xi = [0, 2,   4,   7,   9.5, 11, 12]
yi = [0, 2.8, 4, 5, 6, 9,  12]
# linear interpolation of (xi, yi)
x = np.linspace(xi[0], xi[-1], num_samples)
y = scipy.interpolate.interp1d(xi, yi, kind='quadratic')(x)
# fir filter for smoothing
# scipy.signal.firwin(numtaps, cutoff, width=None, window='hamming', pass_zero=True, scale=True, nyq=1.0)
taps = scipy.signal.firwin(int(num_samples/5), 0.001, width=None, window='hamming',
                           pass_zero=True, scale=True, nyq=1.0)
y = scipy.signal.filtfilt(taps, 1, y)
#y /= np.amax(np.fabs(y))

N = 4
A = np.vander(xi, N)
At = A.transpose()
cc = (np.linalg.inv(At @ A) @ At) @ yi

y = np.polyval(cc, x)

print(cc)

#np.polyval(p, x)

# m = design_signal(num_samples)

fig = plt.figure(0, figsize=(8, 5), frameon=False)
ax = fig.add_subplot(111)

plt.plot(xi, yi, 'r.', markersize=5)
plt.plot(x, y, 'b')
plt.plot(x, np.polyval(cc, x), 'k')


#plt.savefig('dsb_sc_demodulation.eps', format='eps', bbox_inches='tight')
plt.show()
