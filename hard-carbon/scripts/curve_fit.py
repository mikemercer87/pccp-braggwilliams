# Will the optimisation by simplex (method can be replaced later) subject to a set number of variables being free.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Slider, Button, RadioButtons
from leiva2020 import Two_layer
# from leivacython import Two_layer
from dqdv_proc import DQDV
from scipy.signal import chirp, find_peaks, peak_widths
from scipy.interpolate import interp1d
from scipy.optimize import minimize, basinhopping
from pyswarm import pso
import os
import pandas as pd
from time import sleep,strftime

# import bw_widget


class Curve_Fit():
    def __init__(self,initial_arg_dict,list_to_vary,cap_scale):
        self.R = 8.31 # Molar gas constant
        self.F = 96.485 # Faraday const., kC per mole
        self.T = initial_arg_dict['T'] # T in Kelvin
        self.iteration_count = 0        

        self.mass = 0.9*0.004203 # Mass of carbon active material, in g, 90% loading
        self.arg_dict = initial_arg_dict # Initial arguments in model
        self.free_pars = list_to_vary # Array or true or false, indicating values to vary
        self.cap_scale = cap_scale # theoretical re-scaling factor > 250 mAh/g

        self.all_args = ['E0', 'alpha4', 'beta4', 'alpha1', 'beta1', 'delE','g1','g2','g3','g4','g5','g6','L','Cap','Sviba','Svibb','beta3']     
        self.all_pars_dict = {}
        for arg in self.all_args:
            if arg not in ['Cap']:
                self.all_pars_dict[arg] = self.arg_dict[arg]
            elif arg == 'Cap':
                self.all_pars_dict[arg] = [cap_scale]

        print(self.all_pars_dict)
#        self.adjustable_list = ['E0','beta4','delE','g2','g3','g4','L','Sviba','Svibb']
        self.adjustable_list = ['E0','delE','alpha4','beta4','beta3','Sviba','L']
        self.adjustable_withcap = self.adjustable_list[:]
        self.adjustable_withcap.append('Cap') # Copy of list with cap as an adjustable parameter      
        
        self.params = {k : initial_arg_dict[k][0] for k in self.adjustable_list} # Initial test params for Simplex
        self.params['Cap'] = cap_scale
        
        self.params_array = [self.params[k] for k in self.adjustable_withcap]
        self.params_nocap = {k : initial_arg_dict[k][0] for k in self.adjustable_list} # Initial test params for Simplex       
        self.left_exclusion = 7 # ignore up to this point
        self.right_exclusion = -1
        self.left_dqdv = 20
        self.right_dqdv = -1
        self.assign_directories() # Put all results in here                                
        self.errors = ['M','ocv','dH','dS','d2H','d2S','DVA','dqdv','Total']
        self.errors.extend(self.adjustable_withcap) # Errors includes also the parameters that are varied.

        self.var_list = [[] for _ in range(len(self.errors) + len(self.adjustable_withcap))] # contains list of varied parameters and error values
        self.error_dict = {k : l for k in self.errors for l in self.var_list}
        self.fixed_pars() # Assign to csv all values not changes in fitting.
        self.experimental_data()  # Get all experimental plots.
        self.error_weightings()
        self.particle_swarm_limits()

    def particle_swarm_limits(self):
        self.params_upper = []
        self.params_lower = []        
        self.swarm_interval_upper = {param : self.params[param] + 0.2 for param in self.adjustable_withcap}  # Default upper limit of particle swarm
        self.swarm_interval_lower = {param : self.params[param] - 0.2 for param in self.adjustable_withcap}  # Default lower limit of particle swarm
        if 'L' in self.params:
            self.swarm_interval_upper['L'] = min(self.params['L'] + 0.5, 0.5)
            self.swarm_interval_lower['L'] = max(self.params['L'] - 0.5, 0.2)
        if 'beta4' in self.params:    
            self.swarm_interval_upper['beta4'] = self.params['beta4'] + 0.75
            self.swarm_interval_lower['beta4'] = max(self.params['beta4'] - 0.75, 0.75)
        if 'beta3' in self.params:    
            self.swarm_interval_upper['beta3'] = self.params['beta3'] + 0.5
            self.swarm_interval_lower['beta3'] = max(self.params['beta3'] - 0.5, 1.0)
        if 'g2' in self.params:    
            self.swarm_interval_upper['g2'] = self.params['g2'] + 0.02
            self.swarm_interval_lower['g2'] = self.params['g2'] - 0.02
        if 'g3' in self.params:
            self.swarm_interval_upper['g3'] = self.params['g2'] + 0.02
            self.swarm_interval_lower['g3'] = self.params['g2'] - 0.02
        if 'Cap' in self.params:    
            self.swarm_interval_upper['Cap'] = self.params['Cap'] + 40
            self.swarm_interval_lower['Cap'] = self.params['Cap'] - 40
        if 'Sviba' in self.params:
            self.swarm_interval_upper['Sviba'] = self.params['Sviba'] + 1
            self.swarm_interval_lower['Sviba'] = self.params['Sviba'] - 1            
            
        for param in self.adjustable_withcap:
            self.params_upper.append(self.swarm_interval_upper[param])
            self.params_lower.append(self.swarm_interval_lower[param])
            print(self.params_upper, '= params_upper')

    def error_weightings(self):
        self.ocv_w = 1
        self.dH_w = 4
        self.dS_w = 4
#        self.dS_w = 0
        self.d2H_w = 0.5 #0.5
        self.DVA_w = 0.0
        self.dQdV_w = 0.0 #0.1
        self.d2S_w = 0.5 #0.5
        self.w_dict = {'ocv_w' : [self.ocv_w],'dH_w' : [self.dH_w],'dS_w' : [self.dS_w], 'd2H_w' : [self.d2H_w], 'd2S_w': [self.d2S_w], 'DVA' : [self.DVA_w], 'dQdV_w' : [self.dQdV_w]}
        self.w_df = pd.DataFrame.from_dict(self.w_dict)
        self.w_df.to_csv(self.report_directory + 'err_weightings.csv')        
        
    def assign_directories(self):
        self.entropy_path = '../entropy_data/'
        self.figures_path = '../figures/'
        self.galvan_path = '../galvanostatic/'
        self.optimisation_path = '../optimisation_output/'
        self.experimental_dir = self.entropy_path +'HC_entropy20151020_01/'
#        self.experimental_dir = self.entropy_path +'HC_entropydis_20130620_00/'
        self.datedir = strftime("%Y-%m-%d/")
        self.timestr = strftime("%Y-%m-%d_%H%M%S/")
        self.report_directory = self.optimisation_path + self.datedir + self.timestr
        os.makedirs(self.report_directory, exist_ok=True)

    def fixed_pars(self):
        self.not_varied = {}

        for k in self.all_args:
            self.initial_pars = {k : [v] for v in self.params.values()}
            if k not in self.adjustable_withcap:
                self.not_varied[k] = self.all_pars_dict[k]
        self.ipar_df = pd.DataFrame.from_dict(self.all_pars_dict)
        self.nv_df = pd.DataFrame.from_dict(self.not_varied)        
        self.ipar_df.to_csv(self.report_directory + 'initial_pars.csv')
        self.nv_df.to_csv(self.report_directory + 'notvaried_pars.csv')
        
    def error_appending(self):
        self.error_list = [self.arg_dict['M'],self.ocv_error,self.dH_error,self.dS_error,self.d2H_error,self.d2S_error,self.dva_error,self.dqdv_error,self.rms_error] # Error values from a single iteration.
        self.param_list = [self.params[k] for k in self.adjustable_withcap]
        self.error_list.extend(self.param_list)  # Includes error values and parameters in a single table          
        for n, err in enumerate(self.error_list):
            self.var_list[n].append(err)

    def error_report(self):
        for k,l in zip(self.errors,self.var_list):
            self.error_dict[k] = l
        self.error_df = pd.DataFrame.from_dict(self.error_dict)
        self.sorted_errs = self.error_df.sort_values(by = ['M','Total'],ascending=[False,True])
        
    def experimental_data(self):
        # Gets experimental data and puts in convenient format. May need to modify limits. All units in eV per site.         
        file_list = os.listdir(self.experimental_dir)
        entropy_file = [self.experimental_dir + f for f in file_list if f.endswith('entropy.csv')][0]    
        self.df_expt = pd.read_csv(entropy_file, encoding='latin')
        self.df_expt['Normalised cap [mAh/g]'] = self.df_expt['Charge/Discharge [mAh]']/self.mass
        self.max_cap_expt = self.df_expt['Normalised cap [mAh/g]'].max()
        self.df_expt['x'] = self.df_expt['Normalised cap [mAh/g]'] / self.max_cap_expt
        self.df_expt['OCV'] = self.df_expt['OCV [V]   '] * 1000
        self.df_expt['Entropy'] = self.df_expt['M1 Entropy [J mol-1 K-1]']
        self.df_expt['Enthalpy'] = (- self.F * self.df_expt['OCV'] + self.T * self.df_expt['Entropy'])/self.F
        self.df_expt['Entropy'] = self.df_expt['Entropy'] / (self.F)
        self.df_expt['d2H'] = np.gradient(self.df_expt['Enthalpy'].values)/np.gradient(self.df_expt['x'].values)
        self.df_expt['d2S'] = np.gradient(self.df_expt['Entropy'].values)/np.gradient(self.df_expt['x'].values)        
        dqdv = DQDV(entropy_file)
        dqdv.OCV_to_dQdV()
        self.df_dqdv = dqdv.ocv_df_dis
        self.df_dqdv['dQdV'] = -self.df_dqdv['dQdV'] # Note minus sign required
        self.df_dqdv['DVA'] =  -np.gradient(self.df_expt['OCV'].values)/np.gradient(self.df_expt['x'].values) # Note minus sign not required
        self.df_dqdv['Entropy dS/dx'] = self.df_dqdv['M1 Entropy [J mol-1 K-1]'] * self.T / self.F
        self.df_dqdv['Enthalpy dH/dx'] = -self.df_dqdv['OCV'] + self.df_dqdv['Entropy dS/dx']
        df1 = self.df_dqdv[['x','OCV','dQdV','Entropy dS/dx','Enthalpy dH/dx']]
        df1.to_csv('dQdV_data.csv')
        print(list(self.df_dqdv))

    def simulated_data(self,params):
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

    def interpolation_dqdv(self,y_column):
        # Needs debugging!
#        self.cap_ratio = self.cap_scale /self.max_cap_expt
        self.expt_x_dqdv = self.df_dqdv['x'].iloc[self.left_dqdv:self.right_dqdv].to_numpy() *self.max_cap_expt       
        self.sim_x = self.df_sim['x'].to_numpy() *self.cap_scale # Note: not adjusted for capacity.
        self.df_sim_interp = interp1d(x = self.sim_x, y = y_column, kind = 'cubic',bounds_error=False)

        self.sim_result = self.df_sim_interp(self.expt_x_dqdv)
        return(self.sim_result)        

    def arg_dict_to_params(self):
        for par in self.params_nocap:
            self.params_nocap[par] = self.arg_dict[par]

    def params_to_arg_dict(self):
        for par in self.params_nocap:
            self.arg_dict[par] = self.params_nocap[par]

    def array_conversion(self,params_array):
        # Back and forth conversion to keep arrays nicely organised in simulated data. Performs interpolation.
        self.params_array = params_array
        self.params_nocaparray = params_array[:-1]
#        print(self.params_array)
        for n,k in enumerate(self.adjustable_withcap):
            if k not in ['Cap']:
                self.params_nocap[k] = self.params_array[n]
            elif k == 'Cap':
                self.cap_scale = self.params_array[n] # Capacity adjustment here.               
            self.params[k] = self.params_array[n]
#        print(self.params_array)

        self.params_to_arg_dict()
        self.simulated_data(self.params)  # Update simulation result and converts params into form readable in Simplex
        self.arg_dict_to_params()
        self.params_array = [self.params[k] for k in self.adjustable_withcap]
        self.T = self.arg_dict['T']
        # Here the re-scaling to account for cpacity occurs.
        self.scaling_factor =  self.max_cap_expt / self.cap_scale
        self.ocv_interp = self.interpolation(self.df_sim['VV' + self.suffix] * self.scaling_factor)
        self.dH_interp = self.interpolation(-self.df_sim['dH' + self.suffix] * self.scaling_factor /self.F)
        self.dS_interp = self.interpolation(self.T * self.df_sim['dS' + self.suffix] * self.scaling_factor/(self.F * 1000))
        self.d2S_interp = self.interpolation(self.T * self.df_sim['d2S' + self.suffix]* self.scaling_factor * self.scaling_factor / (self.F * 1000))        
        self.d2H_interp = self.interpolation(self.df_sim['d2H' + self.suffix] * self.scaling_factor * self.scaling_factor/ (self.F))        
#        self.df_sim['dxdmu' + self.suffix] = -np.gradient(self.df_sim['x'],self.df_sim['VV' +self.suffix],edge_order=2)
        self.df_sim['dmudx' + self.suffix] = -1/np.gradient(self.df_sim['VV' +self.suffix],self.df_sim['x'],edge_order=2)
        self.df_sim['dxdmu' + self.suffix] = -1/np.gradient(self.df_sim['x'],self.df_sim['VV' +self.suffix],edge_order=2)                
#        self.dqdv_interp = self.interpolation_dqdv(self.df_sim['dxdmu' + self.suffix])
        self.dqdv_interp = self.interpolation_dqdv(self.df_sim['dmudx' + self.suffix] /(self.scaling_factor*self.scaling_factor))
        self.dva_interp = self.interpolation(self.df_sim['dxdmu' + self.suffix] * (self.scaling_factor*self.scaling_factor))                
        self.no_points = len(self.ocv_interp)
        self.no_dqdv = len(self.dqdv_interp)

    def fitting_error_nodQdV(self,params_array):
        # Quantify as weighted function. Sum of OCV, -dH and TdS
        self.ocv_error = np.sqrt(np.sum((self.ocv_interp - self.df_expt['OCV'].iloc[self.left_exclusion:self.right_exclusion])**2)) / self.no_points
        self.dH_error = np.sqrt(np.sum((-self.dH_interp - self.df_expt['Enthalpy'].iloc[self.left_exclusion:self.right_exclusion])**2)) / self.no_points
        self.dS_error = np.sqrt(np.sum((self.dS_interp - self.T * self.df_expt['Entropy'].iloc[self.left_exclusion:self.right_exclusion])**2)) / self.no_points
        self.d2H_error = np.sqrt(np.sum((self.d2H_interp - self.df_expt['d2H'].iloc[self.left_exclusion:self.right_exclusion])**2)) / self.no_points
        self.d2S_error = np.sqrt(np.sum((self.d2S_interp - self.T * self.df_expt['d2S'].iloc[self.left_exclusion:self.right_exclusion])**2)) / self.no_points
        self.dva_error = np.sqrt(np.sum((self.dva_interp - self.df_dqdv['DVA'].iloc[self.left_exclusion:self.right_exclusion])**2)) / self.no_points               
        self.rms_error = self.dS_w * self.dS_error + self.dH_w * self.dH_error + self.ocv_w * self.ocv_error + self.d2H_w * self.d2H_error + self.d2S_w * self.d2S_error + self.DVA_w * self.dva_error
        return(self.rms_error)

    def fitting_error_dQdV(self,params_array):
        # Define the weighting in the final fitting error function. Only handles dQ/dV error.
        self.dqdv_error = np.sqrt(np.sum((self.dqdv_interp - self.df_dqdv['dQdV'].iloc[self.left_dqdv:self.right_dqdv])**2)) / self.no_dqdv

        self.rms_error = self.dQdV_w * self.dqdv_error
        return(self.rms_error)

    def fitting_error(self,params_array):
        self.array_conversion(params_array)
        self.iteration_count += 1        
        self.rms_error = self.fitting_error_nodQdV(params_array) + self.fitting_error_dQdV(params_array)  
        if self.iteration_count % 20 == 0:
            print('OCV_error = ', self.ocv_error, ' ; weighted : ', self.ocv_w * self.ocv_error)
            print('dH_error = ', self.dH_error, ' ; weighted : ', self.dH_w * self.dH_error)
            print('d2H_error = ', self.d2H_error, ' ; weighted : ', self.d2H_w * self.d2H_error)        
            print('dS_error = ', self.dS_error, ' ; weighted : ', self.dS_w * self.dS_error)
            print('d2S_error = ', self.d2S_error, ' ; weighted : ', self.d2S_w * self.d2S_error)        
            print('dqdv_error = ', self.dqdv_error, ' ; weighted : ', self.dQdV_w * self.dqdv_error)
            print('dva_error = ', self.dva_error, ' ; weighted : ', self.DVA_w * self.dva_error)            
            print('rms_error = ', self.rms_error)
            print('params_array = ', self.params_array)            
            print('**********Iteration number: ', self.iteration_count, ' ************')
        self.error_appending()
        return(self.rms_error)

    def particle_swarm(self,phase = 0):
        self.arg_dict['M'] = 200
#        result = basinhopping(self.fitting_error,self.params_array,T=0.1,disp=True,niter=100,stepsize=0.5,minimizer_kwargs={'method': 'Nelder-Mead', 'options':{'xatol':self.default_tolerance,'disp':True,'maxfev':self.maxfev}})
#        self.params_upper = [i + 0.5 for i in self.params_array]
#        self.params_lower = [i - 0.5 for i in self.params_array]
#        self.params_upper[-1] = 400
#        self.params_lower[-1] = 300        
        self.params_array,result = pso(self.fitting_error,self.params_lower,self.params_upper,maxiter=200,debug=True,swarmsize=400,minstep=5e-7,minfunc=5e-7, omega=0.75, phip=0.75, phig=0.75)        
#,minimizer_kwargs = {'method' : 'Nelder-Mead','options' : {'xatol':self.default_tolerance,'disp':True,'maxfev':self.maxfev}}        
        self.plot_obj = Plotting(self)
        self.plot_obj.plotting_main(show=False,tag='swarmM_'+str(self.arg_dict['M']))
        self.plot_obj.plotting_dQdV(show=False,tag='swarmM_'+str(self.arg_dict['M']))                   
    
    def simplex(self,phase = 0):
        self.default_tolerance = 5e-4
        self.maxfev = 1500

#        self.particle_swarm()
#        self.left_exclusion = 7 # increase number of data points.
#        self.particle_swarm(1)

        self.plot_obj = Plotting(self)
        self.plot_obj.plotting_main(show=False,tag='init')
        self.plot_obj.plotting_dQdV(show=False,tag='init')    
        for M in [250,300,350,400]:
            self.arg_dict['M'] = M
#        total_error = self.fitting_error(self.params_array)
            result = minimize(self.fitting_error,self.params_array,method='Nelder-Mead',options={'xatol':self.default_tolerance,'disp':True,'maxfev':self.maxfev})

            print('x_array = ', result.x)
            print('params_array = ', self.params_array)
            print('Iterations = ', self.iteration_count)            
            self.default_tolerance = self.default_tolerance / 2
            self.maxfev= self.maxfev + 1000
            self.plot_obj = Plotting(self)
            self.plot_obj.plotting_main(show=False,tag='M_'+str(M)+'_'+str(phase))
            self.plot_obj.plotting_dQdV(show=False,tag='M_'+str(M)+'_'+str(phase))        

        self.error_report()
        self.sorted_errs.to_csv(self.report_directory + 'error_report_{}.csv'.format(phase))
#        self.arg_dict['M'] = 80
#        result = minimize(self.fitting_error,self.params_array,method='Nelder-Mead',options={'xatol':1e-3,'disp':True,'maxfev':500})

class Plotting():
    def __init__(self,curve_fit):
        self.c_f = curve_fit # Inherit from curve_fit object
        self.formatting_all()
        self.output_dir = curve_fit.report_directory
        self.scaling_factor =  self.c_f.scaling_factor  
            
    def formatting_all(self):
        plt.clf()
        mpl.rcParams['lines.linewidth'] = 2

        font = {'size': 16}

        mpl.rc('font', **font)
        mpl.rcParams['xtick.direction'] = 'in'
        mpl.rcParams['ytick.direction'] = 'in'

        prop_cycle = plt.rcParams['axes.prop_cycle']
        self.colors = prop_cycle.by_key()['color']
        self.msize = 1.5 # markersize

    def plotting_main(self,save=True,show=False,text=True,tag=''):
        self.formatting_all()
        f, (ax1, ax2,ax3) = plt.subplots(3,1, figsize=(6,9),sharex='col')
        axes = (ax1, ax2, ax3)
        self.x_dim, self.y_dim = f.get_size_inches()
 
        l_lim = self.c_f.left_exclusion
        r_lim = self.c_f.right_exclusion
        print('left_exclusion=',l_lim)
 #       self.c_f.expt_x = self.c_f.df_expt['Normalised cap [mAh/g]'].iloc[l_lim:r_lim].to_numpy()
        x = self.c_f.expt_x
     
        x_all = self.c_f.df_expt['Normalised cap [mAh/g]'].iloc[1:]
        df_e = self.c_f.df_expt
        T = self.c_f.T
        # Raw variables are used to plot grey dashed lines: require capacity rescaling.
        xrawsim = self.c_f.df_sim['x'] * self.c_f.cap_scale
        Vrawsim = self.c_f.df_sim['VV' + self.c_f.suffix] * self.scaling_factor
        dHraw = self.c_f.df_sim['dH' + self.c_f.suffix] * self.scaling_factor /self.c_f.F
        d2Hraw = self.c_f.df_sim['d2H' + self.c_f.suffix] * self.scaling_factor * self.scaling_factor /self.c_f.F        
        dSraw = self.c_f.df_sim['dS' + self.c_f.suffix] * self.scaling_factor /(self.c_f.F * 1000)
        
        ax1.plot(xrawsim,Vrawsim,label='Sim,OCV',linestyle=':',color='grey')

        print('len(x) = ', len(x))
        print('len(df_e) =', len(df_e['OCV']))           
        ax1.plot(x,self.c_f.ocv_interp,label='Sim,OCV (fitted)')
        ax1.plot(x_all,df_e['OCV'].iloc[1:],linestyle='',marker='o',label='Expt,OCV',color='grey',markersize=self.msize)
        ax1.plot(x,df_e['OCV'].iloc[l_lim:r_lim],linestyle='',marker='o',label='Expt,OCV (fitted)',color='r',markersize=self.msize)
        
        ax2.plot(xrawsim,-dHraw,label='Sim,dH',linestyle=':',color='grey')         
        ax2.plot(x,self.c_f.dH_interp,label='Sim,dH (fitted)')
        ax2.plot(x_all,-df_e['Enthalpy'].iloc[1:],linestyle='',marker='^',label='Expt,Enthalpy',color='grey',markersize=self.msize)        
        ax2.plot(x,-df_e['Enthalpy'].iloc[l_lim:r_lim],linestyle='',marker='^',label='Expt,Enthalpy (fitted)',color='r',markersize=self.msize)
#        ax2.plot(xrawsim,d2Hraw,label='Sim,d2H',linestyle=':',color='grey')         
#        ax2.plot(x,self.c_f.d2H_interp,label='Sim,d2H (fitted)')
#        ax2.plot(x_all,df_e['d2H'].iloc[1:],linestyle='',marker='^',label='Expt,Enthalpy d2H',color='grey',markersize=self.msize)        
#        ax2.plot(x,df_e['d2H'].iloc[l_lim:r_lim],linestyle='',marker='^',label='Expt,Enthalpy d2H (fitted)',color='r',markersize=self.msize)
#        ax2.set_ylim([-0.5,5.2])
        
        ax3.plot(xrawsim,T * dSraw,label='Sim,dS',linestyle=':',color='grey')    
        ax3.plot(x,self.c_f.dS_interp,label='Sim,dS')
        ax3.plot(x_all,T * df_e['Entropy'].iloc[1:],linestyle='',marker='>',label='Expt,Entropy',color='grey',markersize=self.msize)        
        ax3.plot(x,T * df_e['Entropy'].iloc[l_lim:r_lim],linestyle='',marker='>',label='Expt,Entropy (fitted)',color='r',markersize=self.msize)
        
        ax1.set_ylabel('OCV (V)')
        ax2.set_ylabel('-deltaH (eV)')
        ax3.set_ylabel('TdeltaS (eV)')
        ax3.set_xlabel('Capacity (mAh/g)')
        f.subplots_adjust(hspace=0)
        for ax in axes:
            ax.legend(fontsize = 14)
        plt.tight_layout()
        if text:
            initial_x = 0.42
            initial_y = 0.8
            y_spacing = 0.3
            font_size = 12
            errors = [self.c_f.ocv_error,self.c_f.dH_error,self.c_f.dS_error]
            weights = [self.c_f.ocv_w,self.c_f.dH_w,self.c_f.dS_w]
            final_errors = [error * weight for error, weight in zip(errors,weights)]
            for n, (err,w,f_err) in enumerate(zip(errors,weights,final_errors)):
                message = 'Error = ' + '{:.2f}'.format(w) + ' * ' + '{:.4f}'.format(err) + ' = ' + '{:.4f}'.format(f_err)
                print(message)
                if n < 2:
                    plt.text(x=initial_x,y =initial_y - n*y_spacing, s = message, transform = f.transFigure,fontsize = font_size)
                else:
                    plt.text(x=initial_x,y =initial_y - n*y_spacing + 0.1, s = message, transform = f.transFigure,fontsize = font_size)                    
        
        if save == True:
            plt.savefig(self.output_dir +'main_output_'+ tag + '.png',dpi=600)
        if show:    
            plt.show()

    def plotting_dQdV(self,save=True,show=False,text=True,tag=''):
        self.formatting_all()
        f = plt.figure(figsize=(7,8))
        self.x_dim, self.y_dim = f.get_size_inches()

        font = {'size': 28}        
        l_lim = curve_fit.left_dqdv
        r_lim = curve_fit.right_dqdv
        expt_df = self.c_f.df_dqdv
        raw_ocv = self.c_f.df_sim['VV' + self.c_f.suffix] * self.scaling_factor
        raw_dvdq = (self.c_f.df_sim['dmudx' + self.c_f.suffix] / ( self.scaling_factor * self.scaling_factor))
        ocv_interp_dqdv = self.c_f.interpolation_dqdv(raw_ocv) # Onto same number of points
#    x_interp_dqdv = curve_fit.interpolation_dqdv(curve_fit.df_sim['x']) # Onto same number of points    
#    plt.plot(curve_fit.expt_x,curve_fit.dqdv_interp,linestyle='-',marker='^',label='Model') # Check!
        plt.plot(raw_ocv, raw_dvdq, linestyle=':', color='k', label='Sim,dQdV raw')
        plt.plot(ocv_interp_dqdv, self.c_f.dqdv_interp, linestyle='-', label='Sim,dQdV interpolated') # Check!        

        plt.plot(expt_df['OCV'],expt_df['dQdV'],linestyle='',marker='o',label='Expt,dQdV (all)',markersize=self.msize,color='grey')  
        plt.plot(expt_df['OCV'].iloc[l_lim:r_lim],expt_df['dQdV'].iloc[l_lim:r_lim],linestyle='',marker='o',label='Expt,dQdV (fitted)',markersize=self.msize,color='r')  

        plt.xlabel('Voltage (V)')
        plt.ylabel('dQdV')
        plt.legend(fontsize = 13)
        if text:
            initial_x = 0.4
            initial_y = 0.45
            font_size = 14
            err = self.c_f.dqdv_error
            w = self.c_f.dQdV_w
            f_err = self.c_f.dqdv_error * self.c_f.dQdV_w
            message = 'Error = ' + '{:.3f}'.format(w) + ' * ' + '{:.4f}'.format(err) + ' = ' + '{:.4f}'.format(f_err)
            print(message)
            plt.text(x=initial_x,y =initial_y, s = message, transform = f.transFigure,fontsize = font_size)
        plt.xlim([-0.02,0.42])
        
        plt.tight_layout()
        if save == True:
            plt.savefig(self.output_dir +'dQdV_output_' + tag + '.png',dpi=600)
        if show:
            plt.show()
  
        
if __name__ == '__main__':
    arg_dict = {}
    arg_dict['E0'] = [-0.349]
    arg_dict['delE'] = [-0.329]
    arg_dict['g1'] = [0.0]
#    arg_dict['g2'] = [0.04918]
#    arg_dict['g3'] = [-0.05882]
    arg_dict['g2'] = [0.0]
    arg_dict['g3'] = [0.0]    
    arg_dict['g4'] = [0.0]
    arg_dict['g5'] = [0]
    arg_dict['g6'] = [0]            
    arg_dict['alpha4'] = [-0.700]   
#    arg_dict['alpha1'] = [-0.03815]
    arg_dict['alpha1'] = [0]
    arg_dict['beta4'] = [1.973]
    arg_dict['beta1'] = [17.947]    
       
#    arg_dict['beta3'] = [1.5]
    arg_dict['beta3'] = [1.602]
    arg_dict['T'] = [283.0]
    arg_dict['M'] = 100 # Initial value
    arg_dict['L'] = [0.358]
    arg_dict['label'] = 'L'
    arg_dict['nargs'] = 1
#    arg_dict['Sviba'] = [13.552]
#    arg_dict['Svibb'] = [1.526]
#    arg_dict['Sviba'] = [4.885]
    arg_dict['Sviba'] = [4.844]
    arg_dict['Svibb'] = [0]       
    initial_cap = 328.66
    T = arg_dict['T'][0] # T as float, in K
    unused_args = ['alpha3']
    for arg in unused_args:  # Pad with zeros
        arg_dict[arg] = [0]

    curve_fit = Curve_Fit(arg_dict,None,initial_cap)
    initial_params_arr = curve_fit.arg_dict_to_params()        
    curve_fit.experimental_data()
    curve_fit.simulated_data(initial_params_arr)

#    dqdv_interp = c_f.interpolation_dqdv(c_f.df_sim['dxdmu' + c_f.suffix])
#    print(dqdv_interp)
    
    left_dqdv = curve_fit.left_dqdv
    right_dqdv = curve_fit.right_dqdv

    curve_fit.fitting_error(curve_fit.params_array)

    initial_params_arr = curve_fit.arg_dict_to_params()
    curve_fit.dQdV_w = 0.1
    curve_fit.experimental_data()    
    curve_fit.simulated_data(curve_fit.params_array)
    curve_fit.array_conversion(curve_fit.params_array)
    curve_fit.var_list = [[] for _ in range(len(curve_fit.errors) + len(curve_fit.adjustable_withcap))] # contains list of varied parameters and error values
    curve_fit.error_dict = {k : l for k in curve_fit.errors for l in curve_fit.var_list}
#    print('errors =', curve_fit.errors, 'length :', len(curve_fit.errors))
#    print('var_list =', curve_fit.var_list, 'length :', len(curve_fit.var_list))
#    print('with_cap =', curve_fit.adjustable_withcap, 'length :', len(curve_fit.adjustable_withcap))    
    
#    curve_fit.simplex(phase = 2)
        
#    plotting = Plotting(curve_fit)
#    plotting.plotting_main(save=False,show=True)
#    plotting.plotting_dQdV(save=False,show=True)    


'''
    curve_fit.particle_swarm(phase = 0)
    curve_fit.simplex(phase = 1)
    if 'g2' not in curve_fit.adjustable_list:
        curve_fit.adjustable_list.append('g2')
        curve_fit.adjustable_withcap.append('g2')
        curve_fit.params_array.append(curve_fit.arg_dict['g2'][0])
        curve_fit.var_list.append('g2')
        curve_fit.errors.append('g2')        
    if 'g3' not in curve_fit.adjustable_list:
        curve_fit.adjustable_list.append('g3')
        curve_fit.adjustable_withcap.append('g3')
        curve_fit.params_array.append(curve_fit.arg_dict['g3'][0])
        curve_fit.var_list.append('g3')
        curve_fit.errors.append('g3')        
'''




