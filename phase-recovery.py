# MIT License

# Copyright (c) 2019 Yoshiki Masuyama

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Reference
# [1] Y. Masuyama, K. Yatabe and Y. Oikawa,
# "Griffin-Lim like phase recovery via alternating direction method of
# multipliers," IEEE Signal Process. Lett., vol.26, no.1, pp.184--188, Jan. 2019.

# [2] D. Griffin and J. Lim,
# "Signal estimation from modified short-time Fourier transform,"
# IEEE Trans. Acoust., Speech, Signal Process., vol. 32, no. 2, pp. 236--243,
# Apr. 1984.

# [3] J. Kominek, and A. W. Black, "The CMU Arctic Speech Databases,"
# in Proc. 5th ISCA Speech Synthesis Workshop (SSW5), June 2004. pp. 223--224.


import numpy as np
import librosa
import librosa.display


# Load
fname = 'target.wav'
try:
    data, fs = librosa.load(fname, sr=None)

except FileNotFoundError:
    print('Please upload an audio file "target.wav".')


# Define STFT
winLen = 2**9
shiftLen = 2**8


def STFT(x):
    return librosa.core.stft(x, winLen, shiftLen, winLen)


def iSTFT(x):
    return librosa.core.istft(x, shiftLen, winLen)


def stft_zero_padd(data):
    lf = len(data)
    T = int(np.ceil((lf - winLen) / shiftLen))
    lf_new = winLen + T * shiftLen
    data = np.concatenate([data, np.zeros(lf_new-lf,)])
    return data, lf_new


data, lf = stft_zero_padd(data)

c = STFT(data)
amp = np.abs(c)


# GLA and the proposed algorithm
def mysign(x):
    return np.exp(1j * np.angle(x))


# GLA
def gla(z, amp, max_iter=10):
    for i in range(max_iter):
        x = amp*mysign(z)
        z = STFT(iSTFT(x))
    return x


# Proposed algorithm
def admm_gla(z, u, amp, max_iter=10, rho=0.1):
    for i in range(max_iter):
        x = amp * mysign(z - u)
        v = x + u
        z = (rho * v + STFT(iSTFT(v))) / (1 + rho)
        u = u + x - z
    return x


# Initialization
np.random.seed(seed=0)
spec_shape = amp.shape
z0 = amp * np.exp(2 * np.pi * 1j * np.random.rand(spec_shape[0], spec_shape[1]))
u0 = np.zeros(amp.shape)

x_gla = gla(z0, amp)
x_admm_gla = admm_gla(z0, u0, amp)

# random phase
datar_rand = iSTFT(z0)

# GLA
datar_gla = iSTFT(x_gla)

# Masuyama et al.
datar_admm_gla = iSTFT(x_admm_gla)
