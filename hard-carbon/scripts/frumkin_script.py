import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from scipy.signal import chirp, find_peaks, peak_widths

# Script that plots Frumkin isotherm and determines peak half width, plotting a relationship between half width and repulsive interaction parameter, g.

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

mpl.rc('font', **font)

class Frumkin():
    def __init__(self):
#        self.g_list = [0,20,40,60,80] # units meV
        self.g_list = [i for i in range(-98,101)]
        self.R = 8.3145 # Molar gas constant
        self.F = 96485.33 # Faraday constant
        self.T = 298 # Temperature in Kelvin
        self.E = {} # Will contain all potentials. Key =g, value = potentials
        self.dEdx = {}
        self.dQdV = {}
        self.peak_half_widths=[]
        
    def thermo_variables(self):
        self.x = np.linspace(0.0,1.0,10001)
        self.entropy = -self.R*(np.log(self.x) - np.log(1-self.x))
        for g in self.g_list:
            self.E[g] = -g*(self.x-0.5) / 1000 + self.T * self.entropy / self.F
            self.dEdx[g] = np.gradient(self.E[g]) / np.gradient(self.x)
            self.dQdV[g] = -1/self.dEdx[g]            

    def plot_isotherm(self):
        for g in self.g_list:
#            plt.plot(self.E[g], self.dQdV[g], label = 'g = ' + str(g))
            plt.plot(self.E[g], self.dQdV[g], label = 'g = ' + str(g))            
        plt.xlabel('Voltage')
        plt.ylabel('dQ/dV')
        plt.legend()
        plt.show()

    def half_widths(self):
        for g in self.g_list:        
            peaks, _ = find_peaks(self.dQdV[g])
            results_half = peak_widths(self.dQdV[g], peaks, rel_height=0.5)
            left_index = int(results_half[2])
            right_index = int(results_half[3])
            E_g = self.E[g]
            half_width = -(E_g[right_index] - E_g[left_index]) * 1000
            self.peak_half_widths.append(half_width)
#            print('g='+ str(g)+ ', width ='+ str(results_half[0]) + ', width_heights='+ str(results_half[1]) + ', left_lps=' + str(results_half[2]) + ', right_lps=' + str(results_half[3]))
            print('g='+ str(g)+ ', width ='+ str(half_width))

    def plot_widths(self):
        plt.plot(self.g_list,self.peak_half_widths)
        plt.xlabel('g_value / meV per site')
        plt.ylabel('Peak half width / mV')
        plt.show()

if __name__ == '__main__':
    frumkin=Frumkin()
    frumkin.thermo_variables()
    frumkin.half_widths()
    frumkin.plot_widths()
#    frumkin.plot_isotherm()
