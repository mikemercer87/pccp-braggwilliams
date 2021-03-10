import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

R = 8.31 # Molar gas constant.
x = np.linspace(0,1,10001)
delta_eta = 0.02 # spacing on the axes
eta_1_init =0.7 # initial value of eta
epsilon_1_init = 0 # initial value of eta
eta_2_init = 1 # initial value of eta
epsilon_2_init = 0.3 # initial value of eta

'''
def Sconfig(x, eta_val=1, epsilon=0):
#    eta_val = eta(x,eta_0)
    eta_bar = 1 - eta_val
    xpin = (x - epsilon) / (1 - epsilon) # Pinned sites to determine scaling.
    scale = (eta_val - epsilon) / (1 - epsilon)
    S = R*((scale) * np.log(scale) - (xpin) * np.log(xpin) - (eta_val - xpin) * np.log(eta_val - xpin)) * (1- epsilon)
#    S = R*(-(xpin) * np.log(xpin) - (eta_val - xpin) * np.log(eta_val - xpin)) * (1- epsilon)
#    S = R*(-x * np.log(x) - (1 - x) * np.log(1 - x))
    return(S)
'''

'''
def Sconfig(x, eta_val=1, epsilon=0):
#    eta_val = eta(x,eta_0)
    S = R*((eta_val) * np.log(eta_val) - (x) * np.log(x) - (eta_val - x) * np.log(eta_val - x))
#    S = R*(-(xpin) * np.log(xpin) - (eta_val - xpin) * np.log(eta_val - xpin)) * (1- epsilon)
#    S = R*(-x * np.log(x) - (1 - x) * np.log(1 - x))
    return(S)
'''

def Sconfig(x, eta_val=1, epsilon=0):
#    eta_val = eta(x,eta_0)
    Sideal = -R*(x * np.log(x) + (1 -x) * np.log(1-x))
    xpin = (x - epsilon) / (1 - epsilon)
    Sleft = R*(-xpin * np.log(xpin) - (1 - xpin) * np.log(1 - xpin)) *(1-epsilon)
    Sright = R*((eta_val) * np.log(eta_val) - (x) * np.log(x) - (eta_val - x) * np.log(eta_val - x))
    S = (Sleft * Sright) / Sideal
#    S = R*(-(xpin) * np.log(xpin) - (eta_val - xpin) * np.log(eta_val - xpin)) * (1- epsilon)
#    S = R*(-x * np.log(x) - (1 - x) * np.log(1 - x))
    return(S)

def dSconfig(S,x):
    dS = np.gradient(S) / np.gradient(x)
    return(dS)

def sum_ignore_nan(a,b):
    results = np.array(len(a))
    for n, (i,j) in enumerate(zip(a,b)):
        if i == nan:
            result = j
        elif j == nan:
            result = i
        else:    
            result = i + j
        results[n] = result
    return(results)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.45)

S1 = Sconfig(x, eta_1_init, epsilon_1_init) # Initialise with eta=1
dS1 = dSconfig(S1, x) # Main function
S2 = Sconfig(x, eta_2_init, epsilon_2_init) # Solid solution
dS2 = dSconfig(S2, x) # Main function
s1, = plt.plot(x, dS1, lw=2) # Initial plt
s2, = plt.plot(x, dS2, lw=2) # second solid solution
s3, = plt.plot(x, sum_ignore_nan(dS1,S2) / 2, lw=2) # Initial plt


ax.margins(x=0) # Whitespace padding

axcolor1 = 'lightgoldenrodyellow'
axcolor2 = 'darksalmon'
axeta = plt.axes([0.25, 0.35, 0.65, 0.03], facecolor=axcolor1) # Boxes for the slider. Args: left, bottom, width, height.
axepsilon = plt.axes([0.25, 0.30, 0.65, 0.03], facecolor=axcolor1)
axeta2 = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor2) # Boxes for the slider. Args: left, bottom, width, height.
axepsilon2 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor2)

# Calls slider itself. Args: ax, label, initial value, max value.
seta = Slider(axeta, 'SS1,right', 0.0, 1.0, valinit = eta_1_init, valstep = delta_eta) 
setepsilon = Slider(axepsilon, 'SS1,left', 0.0, 1.0, valinit = epsilon_1_init, valstep = delta_eta)
seta2 = Slider(axeta2, 'SS2,right', 0.0, 1.0, valinit = eta_2_init, valstep = delta_eta) 
setepsilon2 = Slider(axepsilon2, 'SS2,left', 0.0, 1.0, valinit = epsilon_2_init, valstep = delta_eta)

# Auto updates values and y-axis scale.
def update(val):
    eta = seta.val
    epsilon = setepsilon.val
    eta2 = seta2.val
    epsilon2 = setepsilon2.val
    S1 = Sconfig(x,eta,epsilon)
    S2 = Sconfig(x,eta2,epsilon2)
    dS1 = dSconfig(S1,x)
    dS2 = dSconfig(S2,x)        
#    S1 = Sconfig(x,eta)    
    s1.set_ydata(dS1)
    s2.set_ydata(dS2)
    s3.set_ydata(sum_ignore_nan(dS1,dS2) / 2)
#    freq = sfreq.val
#    l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    fig.canvas.draw_idle()

seta.on_changed(update)
setepsilon.on_changed(update)
seta2.on_changed(update)
setepsilon2.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04]) # ?
button = Button(resetax, 'Reset', color=axcolor1, hovercolor='0.975')

def reset(event):
    sfreq.reset()
    samp.reset()
button.on_clicked(reset)

plt.show()
