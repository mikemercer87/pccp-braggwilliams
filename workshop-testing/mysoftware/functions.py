import numpy as np

def square(x):
    return x*x

def oneoverr(x):
    return 1/x


def CentralDifference(f,x,h):
    return((f(x+h) - f(x-h)) / (2*h))


for i in range(2, 8):
    h = 10**(-i)
    print(h, abs(CentralDifference(np.sin, 0.0, h) -1.0))
