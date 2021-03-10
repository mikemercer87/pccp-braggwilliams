# Will the optimisation by simplex (method can be replaced later) subject to a set number of variables being free.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from leiva2020 import Two_layer
from dqdv_proc import DQDV
from scipy.signal import chirp, find_peaks, peak_widths
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import os
import pandas as pd
# import bw_widget


class Curve_Fit():
    def __init__(self,initial_arg_dict,list_to_vary,cap_scale):
        self.R = 8.31 # Molar gas constant
        self.F = 96.485 # Faraday const., kC per mole
        self.T = 298 # T in Kelvin 
        self.entropy_path = '../entropy_data/'
        self.figures_path = '../figures/'
        self.galvan_path = '../galvanostatic/'
        self.experimental_dir = self.entropy_path +'entropydischarge_hc_00/'
        self.mass = 0.9*0.004203 # Mass of carbon active material, in g, 90% loading
        self.arg_dict = initial_arg_dict # Initial arguments in model
        self.free_pars = list_to_vary # Array or true or false, indicating values to vary
        self.cap_scale = cap_scale
        self.experimental_data()  # Get all experimental plots.
        self.adjustable_list = ['E0', 'L', 'alpha4', 'beta4', 'delE','g2']
        self.adjustable_withcap = self.adjustable_list[:]
        self.adjustable_withcap.append('Cap') # Copy of list with cap as an adjustable parameter     
        self.params = {k : initial_arg_dict[k][0] for k in self.adjustable_list} # Initial test params for Simplex
        self.params['Cap'] = cap_scale
        self.params_array = [self.params[k] for k in self.adjustable_withcap]
        self.params_nocap = {k : initial_arg_dict[k][0] for k in self.adjustable_list} # Initial test params for Simplex
        self.params_cocaparray = [self.params[k] for k in self.adjustable_list]        
        self.left_exclusion = 3 # ignore up to this point
        self.right_exclusion = -5
        
    def experimental_data(self):
        # Gets experimental data and puts in convenient format. May need to modify limits. All units in eV per site.         
        file_list = os.listdir(self.entropy_path)
        entropy_file = [self.entropy_path + f for f in file_list if f.endswith('entropy.csv')][0]    
        self.df_expt = pd.read_csv(entropy_file, encoding='latin')
        self.df_expt['Normalised cap [mAh/g]'] = self.df_expt['Charge/Discharge [mAh]']/self.mass
        self.df_expt['OCV'] = self.df_expt['OCV [V]   '] * 1000
        self.df_expt['Entropy'] = self.df_expt['M1 Entropy [J mol-1 K-1]']
        self.df_expt['Enthalpy'] = (- self.F * self.df_expt['OCV'] + self.T * self.df_expt['Entropy'])/self.F
        self.df_expt['Entropy'] = self.df_expt['Entropy'] / (self.F)
        dqdv = DQDV()
        dqdv.OCV_to_dQdV()
        self.df_dqdv = dqdv.ocv_df_dis

    def simulated_data(self,params):
#        for k in adjustable_pars:
#            self.params = {k : params[0], 'L' : params[1], 'alpha4': params[2], 'beta4': params[3]} # Customise more cleanly
        for par in self.params_nocap:
            self.arg_dict[par] = self.params_nocap[par]
        self.two_layer = Two_layer(self.arg_dict)  # ACTION: rewrite code to not have to instantiate the class every time           
        self.df_sim = self.two_layer.solution(optimisation = True)
        diff =  self.two_layer.energy_diff
        self.suffix = str('_%.3f' % diff) # For accessing dataframe elements.

    def interpolation(self,y_column):
        # Generate a simulation grid that matches the experimental domain and data point set. Need to treat dq/dV differently.
        self.sim_x = self.df_sim['x'].to_numpy()*self.cap_scale
        self.df_sim_interp = interp1d(x = self.sim_x, y = y_column, kind = 'cubic',bounds_error=False)
        self.expt_x = self.df_expt['Normalised cap [mAh/g]'].iloc[self.left_exclusion:self.right_exclusion].to_numpy()
        self.sim_result = self.df_sim_interp(self.expt_x)
        return(self.sim_result)

    def arg_dict_to_params(self):
        for par in self.params_nocap:
            self.params_nocap[par] = self.arg_dict[par]

    def params_to_arg_dict(self):
        for par in self.params_nocap:
            self.arg_dict[par] = self.params_nocap[par]                       

    def fitting_error(self,params_array):
        # Quantify as weighted function. Sum of OCV, -dH and TdS
        self.params_array = params_array
        self.params_nocaparray = params_array[:-1]
        print(self.params_array)
        for n,k in enumerate(self.adjustable_withcap):
            if k != 'Cap':
                self.params_nocap[k] = self.params_array[n]
            else:
                self.cap_scale = self.params_array[n] # Capacity adjustment here.
            self.params[k] = self.params_array[n]
        self.params_to_arg_dict()
        self.simulated_data(self.params)  # Update simulation result and converts params into form readable in Simplex
        self.arg_dict_to_params()
        self.params_array = [self.params[k] for k in self.adjustable_withcap]
        self.T = self.arg_dict['T']
        self.ocv_interp = self.interpolation(self.df_sim['VV' + self.suffix])
        self.dH_interp = self.interpolation(-self.df_sim['dH' + self.suffix]/self.F)
        self.dS_interp = self.interpolation(self.T * self.df_sim['dS' + self.suffix]/(self.F * 1000))
        self.no_points = len(self.ocv_interp) 
        self.ocv_error = np.sqrt(np.sum((self.ocv_interp - self.df_expt['OCV'].iloc[self.left_exclusion:self.right_exclusion])**2)) / self.no_points
        self.dH_error = np.sqrt(np.sum((-self.dH_interp - self.df_expt['Enthalpy'].iloc[self.left_exclusion:self.right_exclusion])**2)) / self.no_points
        self.dS_error = np.sqrt(np.sum((self.dS_interp - self.T * self.df_expt['Entropy'].iloc[self.left_exclusion:self.right_exclusion])**2)) / self.no_points        
        self.rms_error = 8 * self.dS_error + self.dH_error + self.ocv_error
        print('OCV_error = ',self.ocv_error)
        print('dH_error = ',self.dH_error)
        print('dS_error = ',8*self.dS_error)
        print('RMS_error = ',self.rms_error)
        return(self.rms_error)

    def simplex(self):
        for M in [60,80,100,120,140,160]:
            self.arg_dict['M'] = M
#        total_error = self.fitting_error(self.params_array)
            result = minimize(self.fitting_error,self.params_array,method='Nelder-Mead',options={'xatol':5e-2,'disp':True,'maxfev':200})
            print('x_array = ', result.x)
            print('params_array = ', self.params_array)
#        self.arg_dict['M'] = 80
#        result = minimize(self.fitting_error,self.params_array,method='Nelder-Mead',options={'xatol':1e-3,'disp':True,'maxfev':500})        
        
if __name__ == '__main__':
    arg_dict = {}
    arg_dict['E0'] = [-0.193]
    arg_dict['delE'] = [-0.182]
    arg_dict['g2'] = [-0.020]
    arg_dict['alpha4'] = [-0.988]
    arg_dict['beta4'] = [1.5]
    arg_dict['T'] = [298.0]
    arg_dict['M'] = 100
    arg_dict['L'] = [0.266]
    arg_dict['label'] = 'L'
    initial_cap = 700
    T = arg_dict['T'][0] # T as float, in K
    unused_args = ['g1','alpha3','beta3','alpha1','beta1']
    for arg in unused_args:  # Pad with zeros
        arg_dict[arg] = [0]    

    curve_fit = Curve_Fit(arg_dict,None,initial_cap)
    curve_fit.experimental_data()
#    curve_fit.simulated_data()
    curve_fit.fitting_error(curve_fit.params_array)
    curve_fit.simplex()
    left_limit = curve_fit.left_exclusion
    right_limit = curve_fit.right_exclusion    
    plt.plot(curve_fit.expt_x,curve_fit.ocv_interp,label='Sim,OCV')
    plt.plot(curve_fit.expt_x,curve_fit.df_expt['OCV'].iloc[left_limit:right_limit],linestyle='',marker='o',label='Expt,OCV')
    plt.plot(curve_fit.expt_x,curve_fit.dH_interp,label='Sim,dH')
    plt.plot(curve_fit.expt_x,-curve_fit.df_expt['Enthalpy'].iloc[left_limit:right_limit],linestyle='',marker='^',label='Expt,Enthalpy')
    plt.plot(curve_fit.expt_x,curve_fit.dS_interp,label='Sim,dS')
    plt.plot(curve_fit.expt_x,T * curve_fit.df_expt['Entropy'].iloc[left_limit:right_limit],linestyle='',marker='>',label='Expt,Entropy')        
    plt.legend()
    plt.show()    
    
