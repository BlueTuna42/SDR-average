from operator import index

import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import random

from numpy.ma.testutils import approx

#============== Parameters =====================================
K = 4 #number of curves to fit

xmax = 1421.08e6    # frequency max plot limit
xmin = 1419.4e6     # frequency min plot limit

ymax = 7.5e-5       # amplitude max plot limit
ymin = 3.8e-5       # amplitude min plot limit

#===============================================================
def sum_(x):
    res = 0
    for el in x:
        res = res + el
    return res

def sumar(x):
    res = np.zeros([1, len(x[0])])
    for el in x:
        for i in range(len(res)):
            res[i] = res[i] + el[i]
    return res

def gauss(x, args_):
    q = ((args_[1] * 2 * np.pi) ** -0.5) * (np.e ** (-((x - args_[0]) ** 2) / (2 * args_[1])))
    return ((args_[1]*2*np.pi)**-0.5)*(np.e**(-((x - args_[0])**2)/(2*args_[1])))

#=================================================
f = open("OutputData.txt", "r")

data_x = []
data_y = []
f.readline()
data = []
for i in range(0, 2800):
    a, b = map(float, f.readline().split())
    data_x.append(a)
    data_y.append(b)

f.close()

xminI = data_x.index(xmin)
xmaxI = data_x.index(xmax)

data_x = data_x[xminI:xmaxI]
data_y = data_y[xminI:xmaxI]

data_x = np.asarray(data_x)
data_y = np.asarray(data_y)

data_x = data_x/1e6
data_y -= min(data_y)

#plt.plot(data_x, data_y,linewidth=1,c='#1663be')

#================================================================

args = [] #[mean, sigmasq, P]
for i in range(K):
    args.append([random.uniform(float(data_x[0]), float(data_x[-1])),random.uniform(0,1),1/K])


b = np.zeros([K, len(data_x)], dtype=float)
N = len(data_x)

for num in range(300):
    for i in range(N):
        b_den = 0.
        for j in range(K):
            b_den = b_den + gauss(data_x[i], args[j])*args[j][2]
        for k in range(K):
            b[k][i] = gauss(data_x[i], args[k])*args[k][2]/b_den
    for k in range(K):
        m_num = 0.
        m_den = 0.
        for i in range(N):
            m_num = m_num + b[k][i] * data_y[i] * data_x[i]
            m_den = m_den + b[k][i] * data_y[i]
        args[k][0] = m_num/m_den
        
        s_num = 0.
        for i in range(N):
            s_num = s_num + b[k][i] * data_y[i] * (data_x[i] - args[k][0])**2
        args[k][1] = (s_num/m_den)
        
        args[k][2] = sum_(b[k]*data_y)/N


y = np.zeros([K, N], dtype = float)

plt.plot(data_x, data_y, linewidth=2)

approx = [0] * N
for k in range(K):
    y[k] = gauss(data_x, args[k])*args[k][2]
    approx += y[k]
    plt.plot(data_x, y[k], linewidth=2)

plt.plot(data_x, approx, c='black')

plt.grid(visible=True)
plt.show()

