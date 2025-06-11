import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import fft, fftfreq, fftshift
import matplotlib.pyplot as plt
import scipy as sc
import random

#============ Parameters =======================
cf = 1419.9e6       # central frequency is 1420.2 MHz
resolution = 5000   # resolution in Hz
nAverage = 200000   # number of samples to average
medianN = 20        # number of samples for median

K = 3               # Number of curves to fit

xmax = 1422.00e6    # frequency max plot limit
xmin = 1419.4e6     # frequency min plot limit

ymax = 4.1e-12      # amplitude max plot limit
ymin = -2e-12      # amplitude min plot limit
#===============================================

def median(arr, n):
    N_0 = len(arr)
    res_med = np.zeros(N_0)

    for i in range(n//2, N_0-n//2):
        res_med[i] = np.median(arr[i-n//2:i+n//2])

    for i in range(-n//2+1, n//2+1):
        res_med[-i] = arr[-i]

    return res_med

#===========================================================

fs, data = wav.read("HackrfCas_1419-9MHz_2.5MHz_20MSPS.wav", mmap=True)

T = 1/fs # sample spacing
N = int((1/resolution) / T) # FFT bin size

print("Sample rate %f MS/s" % (fs / 1e6))
print("Num samples %d" % len(data))
print("Duration: %f s" % (T * len(data)))
iq = np.empty(N, np.complex64)

Nd = N
Td = T
print("Decimated sample rate: %f S/s" % (1 / Td))

#============================ FFT MAGIC =================================
f = np.arange(fs/-2.0+cf, fs/2.0+cf, fs/N) # start, stop, step
xf = fftshift(fftfreq(Nd, Td)) # FFT bin frequencies
yf = np.linspace(0, int(T * len(data)), nAverage) # from 0 to 90 s in 90 steps
zf = np.empty((len(yf), len(xf)), np.float64) # matrix for FFT magnitudes

for i in range(0, len(yf), 1):
    index = int(yf[i] * fs)
    iq.real, iq.imag = data[index:index+N,0], data[index:index+N,1]

    PSD = np.abs(np.fft.fft(iq)) ** 2 / (N * fs)
    #PSD_log = 10.0 * np.log10(PSD)
    zf[i] = np.fft.fftshift(PSD)

spectra = np.mean(zf, 0)
spectraMed = median(spectra, medianN)

#========== REMOVE BACKGROUND =============
file = open("HackRFbackground.txt", "r")

backgr = []
file.readline()
for i in range(0, int((1 / Td)/resolution)):
    x, y = map(float, file.readline().split())
    backgr.append(y)

file.close()

spectra -= backgr
spectraMed -= backgr

#========== SAVE PLOT ============
file = open("OutputData.txt", "w")

file.write(str(cf) + " " + str(resolution) + " " + str(nAverage) + " " + str(xmax) + " " + str(xmin) + " " + str(ymax) + " " + str(ymin) + "\n")

for i in range(len(spectra)):
    file.write(str(f[i]) + " " + str(spectraMed[i]) + "\n")

file.close()

#========================== PLOT ====================================
plt.plot(f, spectra, c='black')
plt.plot(f, spectraMed, c='red')
#plt.plot([xmin, xmax], [4.72e-5, 4.72e-5], c='blue', linewidth=3)

plt.xlabel("Frequency [Hz]")
plt.ylabel("Relative Flux")

plt.grid(True)

plt.ylim(ymin, ymax)
plt.xlim(xmin, xmax)

plt.show()
