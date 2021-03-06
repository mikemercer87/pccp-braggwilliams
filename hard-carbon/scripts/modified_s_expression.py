# Uses the expression in Fultz's paper on zeolites to determine the entropy due to reduced sites in pores (2D nucleation)?

import numpy as np
import matplotlib.pyplot as plt

etas= [1.0,0.8,0.7] # test values. eta =  fraction of available sites, less than or equal 1
R = 8.31 # Molar gas constant.

def eta(x,eta_0):
    eta_val = eta_0 + (1 - eta_0) * x**3
    return(eta_val)

def Sconfig(eta_0,x):
    eta_val = eta(x,eta_0)
    S = R*(eta_val * np.log(eta_val) - x * np.log(x) - (eta_val - x) * np.log(eta_val - x))
#    S = R*(-x * np.log(x) - (1 - x) * np.log(1 - x))
    return(S)

def dSconfig(S,x):
    dS = np.gradient(S) / np.gradient(x)
    return(dS)

x = np.linspace(0,1,1001)
for eta_0 in etas:
    S = Sconfig(eta_0,x)
    dS = dSconfig(S,x)
    plt.plot(x,dS, label= str(eta_0))

plt.legend()
plt.xlabel('x')
plt.ylabel('S')

plt.show()
    
