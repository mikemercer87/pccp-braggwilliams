'''
This is my code which replicates the two layer model of graphite as presented in the papers of Ezequiel et al.. Terms are defined here using the same nomenclature as in that paper, and the energy units are in kT.

To run, the code requires the latest version of Python 3 as well as the follow package dependencies, which can be obtained via pip. Alternatively you can install Anaconda which should have these packages included already, although I have had issues with some of the bundled packages not being up to date. Third option is miniconda, which is more lightweight than Anaconda, which has an awful lot of bloat that you'll never use. Up to you:

numpy
scipy
pandas
matplotlib (plotting scripts only; these can be deactivated)

plotting_2l is an external script that I have written to do the plotting. I will attach and document this separately.

There is an extension in the energy expression which allows all the terms to vary exponentially with the occupation, x. These are defined in the energy expressions: each term has an amplitude alpha and a decay constant beta (see the function E for more details).

Python is object orientated; all the important variables and functions are members of the Two_layer class. And the code makes extensive use of Pandas dataframes, which allow easy input/output of csv files in the form of data tables, so basically an automated version of MS Excel. But this may not be an optimal way of doing things for performance....

The code has a modififed Stirling appromixation, when the class is instantiated, which allows the calculation of much larger system sizes than in the papers. This approximation has been tested and is extremely accurate for M > 170. The use of larger system sizes M approximately 300 is essential to describe the small peak in the voltammograms. I believe the code runs up to M = 900 before numerical issues start to occur.

__init__: just tells the class what to do when the class is instantiated, i.e. assign a lot of variables that are used later in the code.

derivative: differentiates a dataframe df.  

logOmega: logarithm of the degeneracy function omega, defined in one of the papers.

E: energy expression. See the comments there for more detail about the alpha and beta parameters.

logsumExp: not a very interesting function, but this is a numerical trick that reduces the numerical values a bit when summing up exponentials. Imported straight from Stack Overflow...

logQ: logarithm of the partition function, Q.

integrate: clunky, horrible code that allows variables within a dataframe to be integrated. Used as little as possible, if at all.

S: calculation of the absolute entropy, S.

mu: calculation of the chemical potential, mu.

dxdmu: variable equivalent to dQ/dV, also having the same dimensions as a voltammogram.

long_var: not very interesting, just a way to assign the columns within the datastructure so they don't get overwritten.

long_var_dict: puts in the dataframes into a Python dictionary, for ease of access.

solution: this essentially does everything: runs through and assigns all the varibles to a dataframe for ease of access to be plotted (could also be exported to a csv text file with ease)

There are a few prototype, not fully tested, or obsolete modules that are commented out. 

In __main__, the interaction energy arguments are entered on the command line. There are default arguments, so the code will run if you just do "python leiva_2step.py". Please refer to the documentation on the argparse module to see how you would enter the arguments. I provide some examples below. 
'''

from __future__ import division
from sys import argv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import exp,log
from scipy.special import factorial, comb
from time import sleep
import argparse
import matplotlib as mpl
import string
from plotting_2l import Plotting

k_B = 1.38064852e-23 # Boltzmann const. 
e = 1.60217662e-19 # ELectronic charge.
R = 8.3144598 # Molar gas constant in SI units
long_dict = {} # Dictionary for iterating over all the plots.

# This is just stuff used in the plotting scripts. Could be taken out if you don't want to use Matplotlib. 
var_list = ['x','dS','S','VkT','VV','mu','dxdmu','dmudx','dH','g1','g2','E0','G','H'] # All the plots
var_names = {'x':'x','dS':'dS/dx','S':'S','VkT':'V/kT','VV':'V/V','mu':r'$\mu$','dxdmu':r'dx/d$\mu$','dmudx':r'd$\mu$/dx','dH':'d$H$/d$x$', 'g1' :'g1 ','g2' : 'g2','E0' : 'e0','G':'G','H':'H'} # Presentation of variables on plots.
units = {'x':'','dS' : '$2Mk$', 'S' : '$2Mk$', 'VkT' : 'kT','VV' : 'V vs. Li', 'mu' : 'eV', 'dxdmu' : 'eV$^{-1}$','dmudx':'a.u.', 'dH' : 'a.u.', 'g1' : 'kT', 'g2' : 'kT','E0':'kT','G':'a.u.','H':'a.u.'} # Units for all the plots.

# plt.style.use('classic')
mpl.rcParams['lines.linewidth'] = 1.5

font = {'size': 16}

mpl.rc('font', **font)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

class Two_layer():
    # Class to do all the calculations for the two layer model.
    def __init__(self,arg_dict):
        self.counter = 0 # Initialises counter for running through g_zero.
        self.df = pd.DataFrame()                    
        self.arg_dict = arg_dict
        for variable in self.arg_dict:
            if not hasattr(self.arg_dict[variable],"__iter__") and variable not in ('label','M'):
                self.arg_dict[variable]=[self.arg_dict[variable]]
        # Organise input parameters from main.
        self.E0 = sorted(self.arg_dict['E0'])
        self.delE = sorted(self.arg_dict['delE'])
        self.g1 = sorted(self.arg_dict['g1'])
        self.g2 = sorted(self.arg_dict['g2'])        
        self.alpha4 = sorted(self.arg_dict['alpha4']) # Amplitude of exponential decay.
        self.beta4 = sorted(self.arg_dict['beta4']) # Decay constant.
        self.T = sorted(self.arg_dict['T'])
        self.M = self.arg_dict['M']
        self.alpha3 = sorted(self.arg_dict['alpha3'])
        self.beta3 = sorted(self.arg_dict['beta3'])
        self.alpha1 = sorted(self.arg_dict['alpha1'])
        self.beta1 = sorted(self.arg_dict['beta1'])
        self.L = sorted(self.arg_dict['L']) # proportion on sublattice 1
        print('L = ',self.L)
#        self.sigma = sorted(self.arg_dict['sigma'])
        self.fact_array = np.empty((10000))
        for j in range(1000):
           if j == 0:
               self.fact_array[j] = 0
           elif j > 160:
               self.fact_array[j] = np.log(np.sqrt(2*np.pi*j)) + j*(np.log(j) - 1)     
           else:
               self.fact_array[j] = log(factorial(j,exact=True))
#           else:
#            if j == 0:
#                self.fact_array[j] = 0

   
        # Use of the modified Stirling approximation. Note that the standard Stirling approximation turns out to not be accurate enough.

        self.log_Mfact = self.M*(np.log(self.M) - 1) +np.log(np.sqrt(2*np.pi*self.M))

        self.lab = self.arg_dict['label']
        self.variable_list = [self.E0,self.delE,self.g1,self.g2,self.alpha4,self.beta4,self.alpha3,self.beta3,self.alpha1,self.beta1,self.L,self.T]
        self.dataframe_dict = {} 
#        self.fact_array=np.array([log(factorial(j)) for j in range(0,1000)])
        self.int_names = {'g1':self.g1 , 'g2':self.g2, 'E0':self.E0}
        self.int_list = ['g1','g2','E0']
        self.int_dict = {}
#    def update_variable_list(self):
#        self.variable_list = [[self.E0],[self.delE],[self.g1],[self.g2],[self.alpha4],[self.beta4],[self.alpha3],[self.beta3],[self.alpha1],[self.beta1],[self.T]]
        
    def derivative(self,df, df_x, df_y, input_key):
        dy = np.zeros((len(df_x)))
#    dy[0:-1] = (np.diff(df_y)/np.diff(df_x))
#    dy[-1] = 'NaN'
        dy = np.gradient(df_y,df_x,edge_order=2)
        long_key = 'd' + input_key
        df[long_key] = dy
        return df[long_key]

    def logOmega(self,j, N):
    # Log of degeneracy function omega.
        if self.M1 > self.M2:
            # Define j as number of sites on M1. By definition, M2 has less than half the sites.
            if N < (self.M1 + self.M2) / 2:
                if N - j <= self.M2:
                    result = np.log(comb(self.M1,j)*comb(self.M2,N-j))                    
                else:
                    result = np.log(comb(self.M1,j-self.M2))                    
                    
            else:
                if self.M2 - N + j <= self.M2:
                    result = np.log(comb(self.M1,j)*comb(self.M2,self.M1+self.M2-N-j))
                else:
                    result = np.log(comb(self.M1,N-j+self.M2))

        elif self.M1 == self.M2:
            # Easy situation when sublattices are balanced.
            if N < self.M1:
                result = np.log(comb(self.M1,j)*comb(self.M2,N-j))
            else:
                result = np.log(comb(self.M1,j)*comb(self.M2,self.M1+self.M2-N-j))
                
        else:
            # Define j as the number of sites on M2
            if N < (self.M1 + self.M2) / 2:
                if N - j <= self.M1:
                    result = np.log(comb(self.M2,j)*comb(self.M1,N-j))
                else:
                    result = np.log(comb(self.M2,j-self.M1))
            else:
                if self.M1 - N + j <= self.M1:
                    result = np.log(comb(self.M2,j)*comb(self.M1,self.M1+self.M2-N-j))               
                else:
                    result = np.log(comb(self.M2,N-j+self.M1))                          
        return result


    def E(self,j, N):
    # Energy expressions, converted to units of kT.
        x = N / (self.M1 + self.M2)
        if self.M1 <= self.M2:
            # Define j as the occupation on M2
            if N < (self.M1 + self.M2)/2:
                self.n1 = (N - j) / self.M1
                self.n2 = j / self.M2
            else: # fraction of n2 vacancies.
                self.n1 = (N - self.M2 + j) / self.M1
                self.n2 = (self.M2 - j) / self.M2               
        else:
            # Switch to use j as occupation on M1
            if N < (self.M1 + self.M2)/2:
                self.n1 = j / self.M1
                self.n2 = (N - j) / self.M2
            else: # fraction of n2 vacancies.
                self.n1 = (self.M1 - j) / self.M1
                self.n2 = (N - self.M1 + j) / self.M2               

#        else:
#            self.n1 = (self.M - j) / self.M
#            self.n2 = (N - self.M + j) / self.M

        # ******Evaluation of the exponential!********
        # This is possibly the most interesting part. The energy expressions are modifed by a term that depends exponentially on the occupation. The '4' terms have been the most extensively tested and are the most promising: these relate to the modification of the Li-C interaction itself during interaction. We can disucss the output separately.
        # There is also the option to modify the other terms as well. Those have not been tested as extensively yet. In the two layer model, to reproduce the peak in voltammogram at low occupation requires very large alpha1 or alpha3, coupled to delta1 and g respectively, possibly unphysically large.
#        self.E0_prime = self.E0 + self.alpha4 * np.exp(-self.beta4 * x)
        self.E1 = self.E0  + self.alpha4 * np.exp(-self.beta4 * self.n1)     

        self.E2 = self.E0 - self.delE # changing the parameter lowers the votage of the majority sublattice
        g1_terms = self.g1 * (self.n1 * self.n1) # Modified to count Li-vacancy pairs.
        E01_terms = self.E1 * self.n1
        g2_terms =  self.g2 * (self.n2 * self.n2) # Modified to count Li-vacancy pairs.
        E02_terms = self.E2 * self.n2
        result1 = (g1_terms + E01_terms) * e * self.M1
        result2 = (g2_terms + E02_terms) * e * self.M2
        result = result1 + result2
        result = np.longdouble(result)            
        return result

  
    def logSumExp(self,ns):
        max = np.max(ns)
        ds = ns - max
        sumOfExp = np.exp(ds).sum()
        return max + np.log(sumOfExp)

    def logQ(self,N):
        if N < (self.M1 + self.M2) / 2:
            logq_result = self.logSumExp(np.array([(self.logOmega(j, N) - self.E(j, N) * self.beta) for j in range(N + 1)]))
        else:
            logq_result = self.logSumExp(np.array([(self.logOmega(j, N) - self.E(j, N) * self.beta) for j in range(self.M1 + self.M2 - N + 1)]))
        result=(np.longdouble(logq_result))
 #   print Omega(j,N)
        return result

    def subocc1(self,j,N):
        # ! Define as the ratio of occupied sites.
        if self.M1 <= self.M2:
            # Define j as the occupation on M2: this might be the inconsistency!
            if N < (self.M1 + self.M2)/2:
                n1 = j / self.M2
            else: # fraction of n2 vacancies.
                n1 = (self.M2 - j) / self.M2           
        else:
            # Switch to use j as occupation on M1
            if N < (self.M1 + self.M2)/2:
                n1 = j / self.M1
            else: # fraction of n1 vacancies.
                n1 = (self.M1 - j) / self.M1
        return n1

    def subocc1_average(self,N):
        if N < (self.M1 + self.M2) / 2:
            partial_q1 = self.logSumExp(np.array([(np.log(self.subocc1(j, N)) + self.logOmega(j, N) - self.E(j, N) * self.beta) for j in range(N + 1)]))
            logq_result = self.logSumExp(np.array([(self.logOmega(j, N) - self.E(j, N) * self.beta) for j in range(N + 1)]))
        else:
            partial_q1 = self.logSumExp(np.array([(np.log(self.subocc1(j, N)) + self.logOmega(j, N) - self.E(j, N) * self.beta) for j in range(self.M1 + self.M2 - N + 1)]))
            logq_result = self.logSumExp(np.array([(self.logOmega(j, N) - self.E(j, N) * self.beta) for j in range(self.M1 + self.M2 - N + 1)]))
#           logq_result = self.logSumExp(np.array([(self.logOmega(j, N) - self.E(j, N) * self.beta) for j in range(self.M1 + self.M2 - N + 1)]))
        result=(np.exp(partial_q1 - logq_result))
 #   print Omega(j,N)
        return result                   

    def integrate(self,df,df_x, df_y, input_key):
        int_key = str(input_key).lstrip('d')
        d_x = float(df_x.iloc[1] - df_x.iloc[0])
        df[int_key] = 0.0
        for j in range(1,len(df_y)):
            y_array=np.array(df_y[1:j+1])
            df[int_key][j] = np.trapz(y = y_array, dx = d_x)
        return(df[int_key])

    def S(self,N,logQ_array):
        if N == 0 or N == self.M1 + self.M2:
            return(0)
        else:
            if N < self.M:
                beta_array = np.array([self.beta*self.E(j, N) for j in range(N + 1)])
                boltzmanns = np.sum([beta_array[j] * np.exp(self.logOmega(j, N) - beta_array[j] - logQ_array[N]) for j in range(N + 1)])
            else:
                beta_array = np.array([self.beta*self.E(j, N) for j in range(self.M1 + self.M2 - N + 1)])
                boltzmanns = np.sum([beta_array[j] * np.exp(self.logOmega(j, N) - beta_array[j] - logQ_array[N]) for j in range(self.M1 + self.M2 - N + 1)]) 
            result= (boltzmanns + logQ_array[N]) # Shifted Q_array index.
            result= (np.longdouble(result) * R)/ (self.M1+self.M2)
            return(result)
    
    def mu(self,df_x, df_logQ, dlogQ_array):
    # Is in eV!
        mu_result = - (k_B * self.T * dlogQ_array) / e
        return(mu_result)

    def G(self):
    # Is in KJ mol-1!
        G_result = (- (k_B * self.T * self.logQ_array) / (e * (self.M1 + self.M2)))*96.485307
        return(G_result)    

    def dxdmu(self,df,df_x,df_mu,dlogQ_array,long_dxdmu):  
        result = (- 1 / (k_B * self.T * self.derivative(df, df_x, dlogQ_array, long_dxdmu))) * e / self.N_max # Units: per joule. Normalised per particle.
        return(result)

    def long_var(self,var,energy_diff):
        return(str(var + '_%.3f' % energy_diff))

    def long_var_dict(self,long_dict,var,energy_diff):
        long_dict.setdefault(var, [])
        long_dict[var].append([self.long_var(var,energy_diff), '{0:.3f}'.format(energy_diff)])
        
    def var_check(self,key,full_list,long_var,Ediff):
        if key == long_var:
            return(Ediff)
        else:
            value = full_list[key][0]
            return(value)
        
    def solution(self):
        # **** Excecutes and assigns all thermodynamic variables.
        for variables in zip(*self.variable_list):
#            self.df=pd.DataFrame()
            self.counter+=1

            self.E0,self.delE,self.g1,self.g2,self.alpha4,self.beta4,self.alpha3,self.beta3,self.alpha1,self.beta1,self.L,self.T = variables # Unpack the list
            self.M1 = int(self.L * self.M) # absolute number on first sublattice
            self.M2 = int((1 - self.L) * self.M)
            while self.M1 + self.M2 != self.M:
                if self.M1 + self.M2 < self.M:
                    if self.M1 > self.M2:
                        self.M2 += 1  #Increment the smallest sublattice to minimise error.
                    else:
                        self.M1 += 1
                else:
                    if self.M1 > self.M2:
                        self.M1 -= 1
                    else:
                        self.M2 -= 1                    
                
            self.N_array = np.array([i for i in range(0, self.M1 + self.M2 + 1)])
            self.N_max = self.M1 + self.M2        
            print('M1 =', self.M1, 'M2 = ',self.M2) 
#        self.log_M1fact = self.M1*(np.log(self.M1) - 1) +np.log(np.sqrt(2*np.pi*self.M1))
            self.log_M1fact = self.fact_array[self.M1]
            self.log_M2fact = self.fact_array[self.M2]        
            
            self.energy_diff = self.arg_dict[self.lab][self.counter-1]
            self.Ediff = self.energy_diff
            self.beta = 1/(k_B * self.T)      
            print(self.int_dict)    
            long_g1 = self.long_var('g1',self.energy_diff)
            long_g2 = self.long_var('g2',self.energy_diff)            
            long_E0 = self.long_var('E0',self.energy_diff)
#            long_sigma = self.long_var('sigma',self.energy_diff)
            self.df[long_g1] = self.g1
            self.df[long_g2] = self.g2
#            self.df[long_sigma] = self.sigma
#            self.df[long_E0] = np.array([self.E0 + self.alpha4 * np.exp(-self.beta4 * self.N_array[i]/self.N_max) for i in range(0, self.M1 + self.M2 + 1)])
            long_S = self.long_var('S', self.energy_diff)
            self.df['N'] = self.N_array
            self.df['x'] = self.N_array / self.N_max

            self.long_logQ = self.long_var('logQ', self.energy_diff)
            long_mu = self.long_var('mu', self.energy_diff)
            long_dxdmu = self.long_var('dxdmu', self.energy_diff)
            long_dmudx = self.long_var('dmudx', self.energy_diff)
            self.Omegaarray = [sum([self.logOmega(j,N) for j in range(0, N + 1)]) for N in range(0, self.M1 + self.M2 + 1)]
            self.logQ_array = np.array([self.logQ(N) for N in range(0, self.M1 + self.M2 + 1)])

#    print Q_array
            self.df['logQ'] = self.logQ_array
            self.n1_avg = np.array([self.subocc1_average(N) for N in range(0, self.M1 + self.M2 + 1)])
            self.df['n1'] = self.n1_avg
            self.df['n2'] = (self.df['x'] - (1 - self.L) * self.df['n1']) / (self.L)
            
            self.df['op'] = (self.df['n1'] - self.df['n2'])
            self.df['Omega'] = self.Omegaarray
#            plt.plot(self.df['x'],self.df['x'],label='Occupation, x', linestyle=':')            
#            plt.plot(self.df['x'],self.df['n1'],label='n1 (inserted sites)',linestyle='dashed')
#            plt.plot(self.df['x'],self.df['n2'],label='n2 (pore sites)',linestyle='dashed')
#            plt.plot(self.df['x'],self.df['op'],label='Order parameter n1 - n2')
#            plt.plot(self.df['x'],self.df['logQ'],label='logQ')
#            plt.plot(self.df['x'],self.df['logQ1'],label='logQ1')
#            plt.legend(fontsize=18)
#            plt.show()

            self.dlogQ_array = self.derivative(self.df, self.df['N'], self.df['logQ'], self.long_logQ)
            long_VkT = self.long_var('VkT', self.energy_diff)
            long_VV = self.long_var('VV', self.energy_diff)            

            self.df[long_mu] = self.mu(self.df['N'], self.df['logQ'], self.dlogQ_array)
            self.df[long_dxdmu] = self.dxdmu(self.df,self.df['N'], self.df[long_mu], self.dlogQ_array, long_dxdmu)
            self.df[long_dmudx] = 1/self.df[long_dxdmu]
            
            self.df[long_VkT] = - (self.df[long_mu] * e) / (k_B * self.T) # in kT!
            self.df[long_VV] = - self.df[long_mu] # Already in V!

            long_dS = self.long_var('dS',self.energy_diff)
            long_dH = self.long_var('dH',self.energy_diff)
            long_x = self.long_var('x',self.energy_diff)
            long_xmob = self.long_var('xmobile',self.energy_diff)
            long_H = self.long_var('H',self.energy_diff)
            long_G = self.long_var('G',self.energy_diff)

            self.df[long_x] = self.df['x']
            self.df[long_S] = np.array([self.S(N, self.logQ_array) for N in range(0, self.M1 + self.M2 + 1)]) # in format plotted in paper.
            self.df[long_dS] = self.derivative(self.df,self.df['x'],self.df[long_S],long_S) # J per mole per K
            
#            self.df[long_dH] = (-self.df[long_mu] + self.T * (self.df[long_dS] * 2 * self.M / R)) / 1000 # Arbitray units!
#            self.df[long_dH] = (- 2.47769418 * self.df[long_VkT]) + (self.T *self.df[long_dS]) / 1000 # in kJ mol-1
#            self.df[long_G] = self.integrate(self.df,self.df['x'],self.df[long_mu],long_G)
            self.df[long_G] = self.G()
#            self.df[long_H] = self.integrate(self.df,self.df['x'],self.df[long_dH],long_H)
            self.df[long_H] = self.df[long_G] + self.T * self.df[long_S] / 1000
            self.df[long_dH] = self.derivative(self.df,self.df['x'],self.df[long_H],long_H) # J per mole per K

            for var in var_list:
                self.long_var_dict(long_dict,var,self.energy_diff)

            self.df.to_csv('../output_2l/all_data_'+'multi_val' + '=' + str(self.energy_diff)+ '.csv')
            self.dataframe_dict['%.3f' % self.energy_diff] = self.df
            print('*********End of iteration*********\n\n')
#        for key,df in self.dataframe_dict.iteritems():
#            ref_df = self.dataframe_dict['400.000']
#            self.integ_phases(ref_df,df,'dxdmu_' + key,'mu_' + key)
#            print 'key=', key, 'limit2=', self.limit2
#            print 'p1=', self.phase1, 'p2=', self.phase2, 'p3=', self.phase3
        # Return the complete dataframe will all variables included.
        return(self.df)

if __name__ == '__main__':
    # This is important if you wish to use the code well. Refer to the details on the "argparse" module for more information. On one line an entire plot, with one or several variables altered at time, can be generated. You need to change the argument "nargs" to alter the number of command line arguments.
    # There have been some strange numerical issues if the command line variables with multiple arguments are not entered from lowest to highest. Until I get to the bottom of this better to enter them from lowest to highest, i.e. "--g -0.5 -0.45 -0.4" for example.
    # You need to change "label" to the name of the varible that will be plotted, in this case "--label g".
    # Example: say you wanted a single plot, where g was varied as described above. You could enter the following on the command line.
    # "python leiva_2step_MM.py --M 300 --T 300 --nargs 3 --g -0.5 -0.45 -0.4 --label g"
    
    parser = argparse.ArgumentParser(description='Put in some energy parameters')
    parser.add_argument('--E0', type=float, help='Adds a point term, in kT', default = [0], nargs='+')
    parser.add_argument('--g1', type=float, help='In plane interaction of minority sublattice', default = [0], nargs='+')
    parser.add_argument('--g2', type=float, help='In plane interaction parameter of majority sublattice', default = [0], nargs='+')    
    parser.add_argument('--delE', type=float, help='Point term separation', default = [0.0], nargs='+')
    parser.add_argument('--loc', type=int, help='Legend_location', default = 0)
    parser.add_argument('--M',type=int, help = 'removable particles in each sublattice', default = 5)
    parser.add_argument('--T',type=float, help = 'temperature in Kelvin', default = [293.0], nargs='+')
    parser.add_argument('--nargs',type= int, help = 'number of arguments', default = 3)
    #parser.add_argument
    parser.add_argument('--label', type=str,help= 'variable to be labelled in plots.', default = 'g1')
    parser.add_argument('--alpha4', type=float, help='amplitude of exponential decay, coupled to E0', default = [0.0], nargs = '+')
    parser.add_argument('--beta4', type=float, help='decay constant, coupled to E)', default = [0.0], nargs = '+')
    parser.add_argument('--alpha3', type=float, help='amplitude of exponential decay, coupled to g', default = [0.0], nargs = '+')
    parser.add_argument('--beta3', type=float, help='decay constant, coupled to g', default = [0.0], nargs = '+')
    parser.add_argument('--alpha1', type=float, help='amplitude of exponential decay, coupled to g', default = [0.0], nargs = '+')
    parser.add_argument('--beta1', type=float, help='decay constant, coupled to g', default = [0.0], nargs = '+')
    parser.add_argument('--L', type=float, help='fraction on sublattice 1', default = [0.5], nargs = '+')    
#    parser.add_argument('--sigma', type =float, help='quadruplet parameter from Bazant.',default=[0.0], nargs = '+')
    
    args = parser.parse_args(argv[1:])
    
    E0 = args.E0 # Not important, just for ease of syntax.
    delE = args.delE
    g1 = args.g1
    g2 = args.g2
    T = args.T
    M = args.M
    alpha4 = args.alpha4
    beta4 = args.beta4
    beta3 = args.beta3
    alpha3 = args.alpha3
    alpha1 = args.alpha1
    beta1 = args.beta1
    L = args.L    
#    sigma= args.sigma
    
    input_pars = ['E0','delE','g1','g2','alpha4','beta4','alpha3','beta3','alpha1','beta1','T','M','L']
    arg_dict = dict(vars(args))
    del arg_dict['loc']
    del arg_dict['nargs']

    for key,value in arg_dict.items():
        if key in input_pars and key != 'M':
            if len(value) == args.nargs:
                var = key

    for key,value in arg_dict.items():
        if key != 'label' and key != 'M':
            while len(value) < args.nargs:
                arg_dict[key].append(value[-1]) # This allows the variables that aren't changes to be specified once and then duplicated between plots/
    
    counter = 0
    print('arg_dict=', arg_dict.keys())
    two_layer_obj = Two_layer(arg_dict) # Instantiates claa.
    df_dict = two_layer_obj.solution() # gets out all the important variables and puts them in a dictionary of dataframes
    print('df_dict=', df_dict.keys())
    print('long_dict=',long_dict.keys())
    plot_obj = Plotting(df_dict,long_dict)
#    plot_obj.int_plotter()
#    plot_obj.plotter(args.loc)
#    plot_obj.double_column_plot(df_dict,long_dict)
#    sub_plotter(df_dict,long_dict,var)
    plot_obj.column_plot() # These are just different variations on plotting scripts.
#    plot_obj.plotter(0)
#    plot_obj.twobytwo()
    plot_obj.plotter(0)    
