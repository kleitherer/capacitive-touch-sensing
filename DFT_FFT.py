"""

"""
import numpy as np
import matplotlib.pyplot as plt

a = 0.8

# this was given to us
w = np.linspace(0, 2*np.pi, 2000, endpoint=False)
Xw = (1-a**2) / (1 - 2*a*np.cos(w) + a**2)
plt.figure()
plt.plot(w, Xw)
plt.xlabel(r'$\omega$ (rad/sample)')
plt.ylabel(r'$X(\omega)$')
plt.title(r'DTFT of $x[n]=0.8^{|n|}$')
plt.grid()
plt.show()

# now take N samples of X(w) in the frequency domain
def ifft_from_samples(N):
    k = np.arange(N)
    wk = 2*np.pi*k/N
    Xk = (1-a**2) / (1 - 2*a*np.cos(wk) + a**2)
    xN = np.fft.ifft(Xk)
    return np.real(xN)

x20  = ifft_from_samples(20)
x100 = ifft_from_samples(100)

plt.figure()
plt.stem(np.arange(20), x20)
plt.xlabel(r'$n$ (samples)')
plt.ylabel(r'$x_{20}[n]$')
plt.title(r'$x_{20}[n]$ vs sample index for N=20')
plt.show()

plt.figure()
plt.stem(np.arange(100), x100)
plt.xlabel(r'$n$ (samples)')
plt.ylabel(r'$x_{100}[n]$')
plt.title(r'$x_{100}[n]$ vs sample index for N=100')
plt.show()