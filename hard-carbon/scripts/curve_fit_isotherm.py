from __future__ import division
from sys import argv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import factorial
from scipy.optimize import fmin, brute, minimize
from scipy.integrate import quad
from scipy.interpolate import interp1d
from time import sleep
import argparse
import string
from leiva3 import *
import os
import re
from uuid import uuid1

class Experimental_Curve():
# Gets the information from the experimental dx/dV curves.
    def __init__(self,input_dir):
        self.expt_dir = input_dir + 'experimental/'
        self.dxdV_files = [string for string in os.listdir(self.expt_dir) if re.search('Bianchini',string) is not None]
        self.entropy_files = [string for string in os.listdir(self.expt_dir) if re.search('entropy',string) is not None]
        self.dxdV_dict = {}
        self.entropy_dict = {}
        for file_ID in self.dxdV_files:
#            self.overlit_key = file_ID.split('_')[0].lstrip('Li').replace(',','.')
#            self.overlit_key = str('%.2f' % ((float(self.overlit_key) - 1) * 100))
            self.overlit_key = '0.00'
            self.df= pd.read_csv(self.expt_dir+file_ID)
            print(list(self.df))            
            self.df['E'] = self.df['potential(V)']

            self.df = self.df[self.df['E'] > 3.85] # Removes bias towards low voltage.
            self.df = self.df[self.df['E'] < 4.25]
#            self.df['x'] = 1 - self.df['x']
            self.dxdV_array = self.df[['E','x']].to_numpy() # dxdV and x as np.array
#            print self.array
#            self.array[:,[0,1]] = self.array[:,[1,0]] # Swap columns
            self.dxdV_dict[self.overlit_key] = self.dxdV_array
        print(list(self.dxdV_dict), 'Expt_keys')
        for file_ID in self.entropy_files:
            self.overlit_key = file_ID.split('_')[0].lstrip('Li').replace(',','.')
            self.overlit_key = str('%.2f' % ((float(self.overlit_key) - 1) * 100))
            self.df= pd.read_csv(self.expt_dir+file_ID)
            self.df['rms_err'] = np.sqrt(self.df['Bestfit Entropy_Lower [J mol-1 K-1]']**2 + self.df['Bestfit Entropy_Upper [J mol-1 K-1]']**2)
            self.entropy_reduced = self.df[self.df['rms_err'] < 5]
            self.entropy_reduced = self.entropy_reduced[self.entropy_reduced['OCV [V]   '] > 3.8]
            self.entropy_array = self.entropy_reduced[['OCV [V]   ','Bestfit Entropy [J mol-1 K-1]','Bestfit Entropy_Lower [J mol-1 K-1]','Bestfit Entropy_Upper [J mol-1 K-1]']].to_numpy()# OCV, entropy, min_err and max_err as np.array
#            print self.array
#            self.array[:,[0,1]] = self.array[:,[1,0]] # Swap columns
            self.entropy_dict[self.overlit_key] = self.entropy_array
            
    def dxdV_result(self):
        return(self.dxdV_dict)
    def entropy_result(self):
        return(self.entropy_dict)

class dQdV_sim():
    def __init__(self,arg_dict,overlit_val,expt_x):
        self.arg_dict = arg_dict
        self.df_dict = g_zero_solution(arg_dict,0)[overlit_val]
        self.params = self.arg_dict['E0'], self.arg_dict['J1'], self.arg_dict['J2'], self.arg_dict['delta']
        self.overlit_val = overlit_val
        print(self.overlit_val)
        self.expt_x = expt_x # Experimental x values.
    def get_keys(self,sub_dict):
        first_key =''
        second_key =''
        for key in sub_dict:
            if (key.split('_')[0] == 'V') and (len(key.split('_')) > 1):
                first_key = key
            elif (key.split('_')[0] == 'x') and (len(key.split('_')) > 1):
                second_key = key
        return [first_key,second_key]
    def result(self,params):
        counter = 0
        self.params = {'E0' : params[0], 'J1' : params[1], 'J2': params[2], 'delta': params[3]}
        for par in self.params:
            self.arg_dict[par] = self.params[par]
        self.df = g_zero_solution(self.arg_dict,0)[overlit_val]
        first_key,second_key = self.get_keys(self.df)
#        print 'first_key=', first_key, 'second_key=', second_key
        self.sim_array = self.df[[first_key,second_key]].to_numpy()
#        print '*****self.expt_array******', self.expt_x, type(self.expt_x), 'length =', len(self.expt_x)
#        print '*****self.sim_array******', self.sim_array_uninterp[:,1], type(self.sim_array_uninterp[:,1]), 'length=', len(self.sim_array_uninterp[:,1])
        return(self.sim_array)

class Entropy_sim():
    def __init__(self,arg_dict,overlit_val,expt_x):
        self.arg_dict = arg_dict
        self.df_dict = g_zero_solution(arg_dict,0)[overlit_val]
        self.params = self.arg_dict['E0'], self.arg_dict['J1'], self.arg_dict['J2'], self.arg_dict['delta']
        self.overlit_val = overlit_val
        self.expt_x = expt_x # Experimental x values.
    def get_keys(self,sub_dict):
        print(sub_dict.keys())
        first_key =''
        second_key =''
        for key in sub_dict:
            if (key.split('_')[0] == 'V') and (len(key.split('_')) > 1):
                first_key = key
            elif (key.split('_')[0] == 'dSmobvib') and (len(key.split('_')) > 1):
                second_key = key
        return [first_key,second_key]
    def result(self,params):
        counter = 0
        self.params = {'E0' : params[0], 'J1' : params[1], 'J2': params[2], 'delta' : params[3]}
        for par in self.params:
            self.arg_dict[par] = self.params[par]
        self.df = g_zero_solution(self.arg_dict,0)[overlit_val]
        print('df_keys=', list(self.df))
        first_key,second_key = self.get_keys(self.df)
#        print 'first_key=', first_key, 'second_key=', second_key.
        self.sim_array = self.df[[first_key,second_key]].to_numpy()
#        print '*****self.expt_array******', self.expt_x, type(self.expt_x), 'length =', len(self.expt_x)
#        print '*****self.sim_array******', self.sim_array_uninterp[:,1], type(self.sim_array_uninterp[:,1]), 'length=', len(self.sim_array_uninterp[:,1])
        
        return(self.sim_array)

class Fitting_Error():
    def __init__(self,expt_curve_dxdV,sim_curve_dxdV,expt_curve_entropy,sim_curve_entropy,overlit_key,unique_out_dir):
        self.expt_curve_dxdV = expt_curve_dxdV
        self.sim_curve_dxdV = sim_curve_dxdV
        self.expt_curve_entropy = expt_curve_entropy
        self.sim_curve_entropy = sim_curve_entropy
        self.objective_list = []
        self.parameter_dict = {'E0':[],'J1':[],'J2':[],'delta':[]}
        self.parameter_names = ['E0','J1','J2','delta']
        self.qprime_list = []
        self.sprime_list = []
#        print 'expt_curve =', self.expt_curve
#        print 'sim_curve = ',self.sim_curve
    def __call__(self,params,nelder_mead = False):
        print('params in call=', params)
#        self.sim_curve=self.sim_curve(params) 
        self.expt_col1_dxdV = self.expt_curve_dxdV[:,0]
        self.expt_col2_dxdV = self.expt_curve_dxdV[:,1]
        self.min_x_dxdV = np.amin(self.expt_col1_dxdV)
        self.max_x_dxdV = np.amax(self.expt_col1_dxdV)
        self.sim_result_dxdV = self.sim_curve_dxdV.result(params)
        self.sim_x_dxdV = self.sim_result_dxdV[:,0]
        self.sim_y_dxdV = self.sim_result_dxdV[:,1]
        self.sim_x_red_dxdV = self.sim_x_dxdV[np.where((self.sim_x_dxdV > 3.85) & (self.sim_x_dxdV < 4.25))]
        self.sim_y_red_dxdV = self.sim_y_dxdV[np.where((self.sim_x_dxdV > 3.85) & (self.sim_x_dxdV < 4.25))]
#        self.sim_x_red = self.sim_x[np.where((self.sim_x > self.min_x) & (self.sim_x < self.max_x))]
#        self.sim_y_red = self.sim_y[np.where((self.sim_x > self.min_x) & (self.sim_x < self.max_x))]
        self.n_points_dxdV = len(self.sim_x_red_dxdV)
#        print 'sim_result=', self.sim_result[:,0]

#        self.expt_col1_red = self.expt_col1[np.where((self.expt_col1 > 0.2) & (self.expt_col1 < 0.8))]
#        self.expt_col2_red = self.expt_col2[np.where((self.expt_col1 > 0.2) & (self.expt_col1 < 0.8))]

#        print 'column1=', self.expt_col1_red
        self.expt_interp_method_dxdV = interp1d(x = self.expt_col1_dxdV, y = self.expt_col2_dxdV, kind = 'cubic',fill_value=0,bounds_error=False)
        self.expt_result_dxdV = self.expt_interp_method_dxdV(self.sim_x_red_dxdV)
#        plt.plot(self.sim_x,self.sim_y,label='sim',linestyle='',marker='o')
#        plt.plot(self.sim_x,self.expt_result,label='expt_interpolated',linestyle='',marker='^')
#        plt.legend(loc=0)
#        plt.show()
     
        if self.n_points_dxdV > 10:
            self.objective_dxdV = np.sqrt(np.sum((self.expt_result_dxdV - self.sim_y_red_dxdV) ** 2)/self.n_points_dxdV)
        else:
            self.objective_dxdV = 999 #Crude handling of edge points.
        print('objective_dxdV=', self.objective_dxdV)
        self.expt_col1_entropy = self.expt_curve_entropy[:,0]
        self.expt_col2_entropy = self.expt_curve_entropy[:,1]
        self.expt_col3_entropy = self.expt_curve_entropy[:,2]
        self.expt_col4_entropy = self.expt_curve_entropy[:,3]
        self.min_x_entropy = np.amin(self.expt_col1_entropy)
        self.max_x_entropy = np.amax(self.expt_col1_entropy)
        self.sim_result_entropy = self.sim_curve_entropy.result(params)
        self.sim_x_entropy = self.sim_result_entropy[:,0]
        self.sim_y_entropy = self.sim_result_entropy[:,1]
        self.sim_x_red_entropy = self.sim_x_entropy[np.where((self.sim_x_entropy > 3.95) & (self.sim_x_entropy < 4.15))]
        self.sim_y_red_entropy = self.sim_y_entropy[np.where((self.sim_x_entropy > 3.95) & (self.sim_x_entropy < 4.15))]

        self.n_points_entropy = len(self.sim_x_red_entropy)

#        print 'sim_result=', self.sim_result[:,0]

#        print 'column1=', self.expt_col1_red
        self.expt_interp_method_entropy = interp1d(x = self.expt_col1_entropy, y = self.expt_col2_entropy, kind = 'cubic',fill_value='extrapolate',bounds_error=False)
        self.expt_result_entropy = self.expt_interp_method_entropy(self.sim_x_red_entropy)
        if self.n_points_entropy > 10:
            self.objective_entropy = np.sqrt(np.sum((self.expt_result_entropy - self.sim_y_red_entropy) ** 2)/self.n_points_entropy)
        else:
            self.objective_entropy = 999
        print('objective_entropy=', self.objective_entropy)
        self.qprime_list.append(4 * self.objective_dxdV)
        self.sprime_list.append(self.objective_entropy)
#        self.overall_objective = self.objective_entropy + 4 * self.objective_dxdV
        self.overall_objective = self.objective_dxdV # Just use the dQ/dV as objective.
        print('overall_objective=', self.overall_objective)
        if nelder_mead != False:
            self.objective_list.append(self.overall_objective)
            for index,key in enumerate(self.parameter_names):
                self.parameter_dict[key].append(params[index])
        return(self.overall_objective)
    def final_objective(self):
        return(self.overall_objective)
    

class Optimiser():
    def __init__(self,arg_dict,overlit_val,input_dir,output_dir,min_vals,max_vals,inc_vals):
        self.arg_dict = arg_dict
        self.overlit_val = overlit_val
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.expt_inst = Experimental_Curve(self.input_dir)
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.inc_vals = inc_vals
    def optimise_dxdV(self):                                                  
        expt_curve_dict = self.expt_inst.dxdV_result()
        expt_curve = expt_curve_dict[overlit_val]
        sim_curve = dQdV_sim(arg_dict,overlit_val,expt_curve[:,0])              
        params = [[E0], [J1], [J2], [delta],[Svib]]
        error = Fitting_Error_dQdV(expt_curve,sim_curve,overlit_val)
        result = minimize(error,params,method='Nelder-Mead',options={'xtol':1e-2,'disp':True,'maxfev':2000})
        optimised_pars = result.x
        print('optimised_pars=', optimised_pars, type(optimised_pars))
        return(optimised_pars)
    def optimise_entropy(self):                                                 
        expt_curve_dict = self.expt_inst.entropy_result()
        expt_curve = expt_curve_dict[overlit_val]
        sim_curve = Entropy_sim(arg_dict,overlit_val,expt_curve[:,0])              
        params = [[E0], [J1], [J2], [delta],[Svib]]
        error = Fitting_Error_entropy(expt_curve,sim_curve,overlit_val,self.output_dir)
        result = minimize(error,params,method='Nelder-Mead',options={'xtol':1e-3,'disp':True,'maxfev':2000})
        optimised_pars = result.x
        print('optimised_pars=', optimised_pars, type(optimised_pars))
        return(optimised_pars)
    def optimise_both(self):                                                 
        expt_curve_dict_dxdV = self.expt_inst.dxdV_result()
        expt_curve_dxdV = expt_curve_dict_dxdV[overlit_val]
        expt_curve_dict_entropy = self.expt_inst.entropy_result()
        expt_curve_entropy = expt_curve_dict_entropy[overlit_val]
        sim_curve_dxdV = dQdV_sim(arg_dict,overlit_val,expt_curve_dxdV[:,0])
        sim_curve_entropy = Entropy_sim(arg_dict,overlit_val,expt_curve_entropy[:,0])
        for i in range(0,3):
            total_error = Fitting_Error(expt_curve_dxdV,sim_curve_dxdV,expt_curve_entropy,sim_curve_entropy,overlit_val,self.output_dir)
            rranges=[slice(self.min_vals[k],self.max_vals[k],self.inc_vals[k]) for k in range(0,4)]
            coarse_result = brute(total_error, ranges=rranges,full_output=True,disp=True,finish=None)
            coarse_pars, coarse_fval, coarse_grid, coarse_all_vals = coarse_result
            qprime_arr = np.array(total_error.qprime_list)
            sprime_arr = np.array(total_error.sprime_list)
            cur_pars= coarse_pars
            for j,(cur_par,old_min,old_max,old_inc) in enumerate(zip(cur_pars,self.min_vals,self.max_vals,self.inc_vals)):
                print('j=', j, ',cur_par=', cur_par, ',old_min=', old_min, ',old_max=',old_max,',old_inc=',old_inc)
                new_min = (cur_par + old_min) / 2
                new_max = (cur_par + old_max) / 2
                new_inc = (new_max - new_min) / 4 # Customise later
                print('new_min=', new_min)
                self.min_vals[j] = new_min
                self.max_vals[j] = new_max
                self.inc_vals[j] = new_inc
            no_coarse_points = int((coarse_grid.size)/4)
            coarse_grid = coarse_grid.reshape(4,no_coarse_points).transpose()
            coarse_all_vals = coarse_all_vals.reshape(1,no_coarse_points).transpose()
            qprime_grid = qprime_arr.reshape(1,no_coarse_points).transpose()
            sprime_grid = sprime_arr.reshape(1,no_coarse_points).transpose()
            print('coarse_grid_shape=',coarse_grid.shape)
            print('qprime_shape=', qprime_grid.shape)
            print('sprime_shape=', sprime_grid.shape)
            print('coarse_vals_shape=',coarse_all_vals.shape)
            print('Pars, 4qprime, Sprime, fval' , coarse_pars, qprime_grid,sprime_grid,coarse_fval, type(coarse_pars), type(coarse_fval))
            coarse_grid_vals = np.hstack((coarse_grid,qprime_grid,sprime_grid,coarse_all_vals))
            with open(self.output_dir + '/coarse_pars_'+str(i)+ '.csv','w') as f:
                f.write('E0,J1,J2,delta,Svib\n' + ','.join([str(par) for par in coarse_pars]))
            with open(self.output_dir + '/coarse_fval_'+str(i)+'.csv','w') as f:
                f.write('4Qprime,Sprime,Total Objective\n'+ ','.join([str(obj) for obj in [qprime_arr[-1],sprime_arr[-1],coarse_fval]]))
            print('coarse_grid_vals', coarse_grid_vals)
            pd.DataFrame(coarse_grid_vals,columns=['E0','J1','J2','delta','4Qprime','Sprime','Total Objective']).to_csv(self.output_dir + '/coarse_grid_vals_'+str(i)+'.csv')
            
        nelder_mead = (True,)
        total_error = Fitting_Error(expt_curve_dxdV,sim_curve_dxdV,expt_curve_entropy,sim_curve_entropy,overlit_val,self.output_dir)
        fine_result = fmin(total_error,coarse_pars,xtol=1e-05,ftol=1e-05,full_output=True,disp=True,retall=True,maxfun=2000,args=nelder_mead)       
        opt_pars = fine_result[0]
        opt_fval = fine_result[1]
        opt_all_pars = fine_result[5]
        self.qprime_arr = np.array(total_error.qprime_list)
        self.sprime_arr = np.array(total_error.sprime_list)                                                        
        no_opt_vals = len(opt_all_pars)
        self.objective_list = total_error.objective_list
        self.parameter_dict = total_error.parameter_dict
        opt_all_pars_columns = np.array((opt_all_pars))
        opt_all_fvals_columns = np.array((self.objective_list))
        print('all_pars_shape =', opt_all_pars_columns.shape)
        print('all_fvals_shape =', opt_all_fvals_columns.shape)
#        opt_all = np.hstack((opt_all_pars_columns,opt_all_fvals_columns))
#        pd.DataFrame(opt_all_pars,columns=['E0','J1','J2','delta','Svib']).to_csv(self.output_dir + '/opt_all_pars.csv')
        with open(self.output_dir + '/opt_fval.csv','w') as f:
            f.write('4Qprime,Sprime,Total Objective\n' + ','.join([str(obj) for obj in [self.qprime_arr[-1],self.sprime_arr[-1],opt_fval]]))
#        pd.DataFrame(opt_all,columns=['E0','J1','J2','delta','Svib','Objective']).to_csv(self.output_dir + '/opt_all.csv')

        self.final_objective = total_error.final_objective()
        return(opt_pars)

if __name__ == '__main__':
    storage_dir = './'
    unique_out_dir = 'isotherm_output/' + str(uuid1())
    parser = argparse.ArgumentParser(description='For energy parameters: enter minimum, maximum and increment.')
    os.mkdir(unique_out_dir)
    parser.add_argument('--E0', type=float, help='Adds a point term', default = [4.09,4.35,0.032], nargs='+')
    optimisable_pars = ['E0','J1','J2','delta']
#    optimisable_pars = ['E0','J1','J2','delta']
    range_pars = ['min','max','inc']
    parser.add_argument('--deltaE', type=float, help='Separation between point terms.', default = [0.0,0.0,0.0], nargs='+')
    parser.add_argument('--J1', type=float, help='Nearest neighbour parameter.', default = [25.0,75.0,5.0], nargs='+')
    parser.add_argument('--J2', type=float, help='Next nearest neighbour parameter.', default = [-2.0,17.0,1.5], nargs='+')
    parser.add_argument('--delta', type=float, help='Separation next nearest neighbours on each sublattice.', default = [0.0,16.0,2.0], nargs='+')
    parser.add_argument('--loc', type=int, help='Legend_location', default = 0)
    parser.add_argument('--overlit',type=int, help = 'Overlithiation values, as percentage', default = [0], nargs='+')
    parser.add_argument('--Mprime',type=int, help = 'particles in each sublattice', default = 100)
    parser.add_argument('--T',type=float, help = 'particles in each sublattice', default = [293.0], nargs='+')
    parser.add_argument('--nargs',type= int, help = 'number of arguments', default = 1)
    parser.add_argument('--label', type=str,help= 'variable to be labelled in plots.', default = 'overlit')
    parser.add_argument('--Svib',type=float,help='Vibrational correction.', default = [-10.0,-2.0,2.0], nargs='+')
#    parser.add_argument('--Svib',type=float,help='Vibrational correction.', default = [0.0],)
            
    args = parser.parse_args(argv[1:])
    arg_dict = dict(vars(args))
    input_var_dict={}
    for par in optimisable_pars:
        for index,rangevar in enumerate(range_pars):
            new_key = par + '_' + rangevar
            input_var_dict[new_key] = [arg_dict[par][index]]
    print('input_var_dict=', input_var_dict)        
    pd.DataFrame(input_var_dict).to_csv(unique_out_dir +'/input_ranges.csv')
    
    E0_min,E0_max,E0_inc = args.E0
#    deltaE = args.deltaE
    J1_min, J1_max,J1_inc = args.J1
    J2_min,J2_max,J2_inc = args.J2
    deltaE_min,deltaE_max,deltaE_inc=args.deltaE
    T = args.T
    Svib_min,Svib_max,Svib_inc = args.Svib
    delta_min,delta_max,delta_inc = args.delta
    overlit = args.overlit
    Mprime = args.Mprime
#    Svib_min,Svib_max,Svib_inc  = args.Svib
    min_vals = [E0_min,J1_min,J2_min,delta_min]
    max_vals = [E0_max,J1_max,J2_max,delta_max]
    inc_vals = [E0_inc,J1_inc,J2_inc,delta_inc]

    del arg_dict['loc']
    del arg_dict['nargs']

    overlit_val = str('%.2f' % overlit[0])
    with open(unique_out_dir + '/overlit_val','w') as f:
        f.write(overlit_val)
    opt_vals = Optimiser(arg_dict,overlit_val,storage_dir,unique_out_dir,min_vals,max_vals,inc_vals)

#    optimised_vals_entropy = Optimiser(arg_dict,overlit_val).optimise_entropy()
    optimised_vals = opt_vals.optimise_both()
    print('optimised_vals=', optimised_vals)
    simplex_objective_list = opt_vals.objective_list
    simplex_objective_vecs = opt_vals.parameter_dict
    simplex_df = pd.DataFrame(simplex_objective_vecs)
    print('size_df=', len(simplex_df), 'size_qarr=', len(opt_vals.qprime_arr))
    simplex_df['4Qprime'] = opt_vals.qprime_arr
    simplex_df['Sprime'] = opt_vals.sprime_arr                                                              
    simplex_df['Total Objective'] = simplex_objective_list                                                                  
    simplex_df = simplex_df[['E0','J1','J2','delta','4Qprime','Sprime','Total Objective']]
    best_pars = simplex_df.iloc[-1]                                                              
    simplex_df.to_csv(unique_out_dir + '/simplex_allvals.csv')
#    objective_of_minimum = opt_vals.final_objective
#    minimum_qprime = opt_vals.qprime_list[-1]
#    minimum_sprime = opt_vals.sprime_list[-1]                                                              
 #   optimised_df = pd.DataFrame(optimised_vals).T
 #   optimised_df.columns = optimisable_pars
    best_pars.to_csv(unique_out_dir + '/opt_pars.csv')
    arg_dict['E0'] = [optimised_vals[0]]
    arg_dict['J1'] = [optimised_vals[1]]
    arg_dict['J2'] = [optimised_vals[2]]
    arg_dict['delta'] = [optimised_vals[3]]
#    arg_dict['Svib'] = [optimised_vals[4]]
#    arg_dict['deltaE'] = [optimised_vals[5]]
    params = [arg_dict['E0'],arg_dict['J1'],arg_dict['J2'],arg_dict['delta']]
#    print 'new_arg_dict=', arg_dict
#    expt_curve = Experimental_Curve().dxdV_result()[overlit_val]
#    new_sim_result = dQdV_sim(arg_dict,overlit_val,expt_curve[:,0]).result(params)
    experimental_inst = Experimental_Curve(storage_dir)
    dQdV_expt_curve = experimental_inst.dxdV_result()[overlit_val]
    dQdV_sim_opt = dQdV_sim(arg_dict,overlit_val,dQdV_expt_curve[:,0]).result(params)
    dQdV_df = pd.DataFrame(dQdV_sim_opt,columns=['E','dx/dE'])
    dQdV_df.to_csv(unique_out_dir + '/dqdv_sim.csv')
    entropy_expt_curve = experimental_inst.entropy_result()[overlit_val]
    entropy_sim_opt = Entropy_sim(arg_dict,overlit_val,entropy_expt_curve[:,0]).result(params)
    entropy_df = pd.DataFrame(entropy_sim_opt,columns=['E','dS'])
    entropy_df.to_csv(unique_out_dir + '/entropy_sim.csv')
    f, ((ax1, ax2)) = plt.subplots(1,2, sharex='col', figsize =(7,4))
    axes=(ax1,ax2)
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax1.set_xlabel('Voltage vs. Li/Li$^{+}$ (V)')
    ax2.set_xlabel('Voltage vs. Li/Li$^{+}$ (V)')
    ax1.set_ylabel('dx/dV (V)')
    ax2.set_ylabel('d$S$/d$x$ (J mol$^{-1}$ K$^{-1}$)')
    ax1.set_xticks(np.arange(3.8,4.4,0.1))
    ax2.set_xticks(np.arange(3.8,4.4,0.1))
    ax1.set_xlim([3.80,4.30])
    ax2.set_xlim([3.80,4.30])
    ax1.errorbar(dQdV_expt_curve[:,0],dQdV_expt_curve[:,1],label='Expt,dxdV',linestyle='',marker='o',markersize=3)
    ax1.plot(dQdV_sim_opt[:,0],dQdV_sim_opt[:,1],label='Opt Fit,dxdV')
    ax2.errorbar(entropy_expt_curve[:,0],entropy_expt_curve[:,1],yerr=[entropy_expt_curve[:,2],entropy_expt_curve[:,3]],label='Expt,entropy',linestyle='',marker='o',markersize=3)
    ax2.plot(entropy_sim_opt[:,0],entropy_sim_opt[:,1],label='Opt Fit,entropy')
    ax1.legend(loc=0,fontsize=8)
    ax2.legend(loc=0,fontsize=8)
    plt.tight_layout()
    plt.savefig(unique_out_dir + '/dxdV_entropy.png')
