# Script to generate plots to replicate Leiva's 2 level model.


from sys import argv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import exp,log,factorial
# from scipy.misc import factorial
from time import sleep
import argparse
import matplotlib as mpl
import string

k_B = 1.38064852e-23 # Boltzmann const. 
e = 1.60217662e-19 # ELectronic charge.
# M = 100 # Number of particles in each of the sublattices.
# N_max = 2 * M # Total number of particles.
# logfact_M=log(factorial(M)) # Used to avoid large numbers.
# fact_Msquared=np.longdouble(fact_M**2) # Optimised to avoid repetition
# fact_array=np.array([min(log(factorial(j)),log(factorial(N_max-j))) for j in range(0,N_max)],dtype=np.longdouble) # Saves recalculating every time.
fact_array=np.array([log(factorial(j)) for j in range(0,1000)])
epsilon = 0 # Arbitrary energy mount point, in eV. Sets the energy scale, not the difference.
R = 8.3144598 # Molar gas constant in SI units
Ediff_list = [0.05,0.03,0.02,0.01,0] # List of deltaE values to iterate over (should be float).
g_list = [0]
long_dict = {} # Dictionary for iterating over all the plots.
# var_list = ['x','xmobile','dS','dSmob','S','V','mu','dxdmu', 'dxmobdmu', 'dH', 'H', 'dG', 'G', 'n1', 'n2'] # All the plots
# var_names = {'x':'x','xmobile':'x$_{r}$','dS':'dS/dx','dSmob':'dS/dx$_{r}$','S':'S','V':'V','mu':r'$\mu$','dxdmu':r'dx/d$\mu$', 'dxmobdmu':r'dx$_{r}$/d$\mu$', 'dH':'dH/dx', 'H':'H', 'dG':'dG/dx', 'G':'G', 'n1':'n1', 'n2':'n2'} # Presentation of variables on plots.
var_list = ['x','xmobile','dS','dSmob','S','V','mu','dxdmu', 'dxmobdmu','dSmobvib'] # All the plots
var_names = {'x':'x','xmobile':'x$_{r}$','dS':'dS/dx','dSmob':'dS/dx$_{r}$','S':'S','V':'V','mu':r'$\mu$','dxdmu':r'dx/d$\mu$', 'dxmobdmu':r'dx$_{r}$/d$\mu$','dSmobvib':'dS/dx$_{r}$'} # Presentation of variables on plots.
# units = {'x':'','xmobile':'','dS' : 'J mol$^{-1}$ K$^{-1}$', 'dSmob' : 'J mol$^{-1}$ K$^{-1}$', 'S' : 'J mol$^{-1}$ K$^{-1}$', 'V' : 'V vs. Li/Li$^{+}$', 'mu' : 'eV', 'dxdmu' : 'eV$^{-1}$', 'dxmobdmu' : 'eV$^{-1}$', 'dH' : 'kJ mol$^{-1}$', 'H' : 'kJ mol$^{-1}$', 'dG' : 'kJ mol$^{-1}$', 'G' : 'kJ mol$^{-1}$', 'n1' : '', 'n2' : ''} # Units for all the plots.
units = {'x':'','xmobile':'','dS' : 'J mol$^{-1}$ K$^{-1}$', 'dSmob' : 'J mol$^{-1}$ K$^{-1}$','dSmobvib': 'J mol$^{-1}$ K$^{-1}$', 'S' : 'J mol$^{-1}$ K$^{-1}$', 'V' : 'V vs. Li/Li$^{+}$', 'mu' : 'eV', 'dxdmu' : 'eV$^{-1}$', 'dxmobdmu' : 'eV$^{-1}$'} # Units for all the plots.
Nfirst = 2 # Number of nearest neighbours in the lattice. 
Nsecond = 6 # Number of second nearest neighbours (double counting accounted later)

plt.style.use('classic')
mpl.rcParams['lines.linewidth'] = 1.5

font = {'size': 13}

mpl.rc('font', **font)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

def deltaV(df,input_key):
    volt_name = 'V_' + input_key
    out_name = 'dxdmu_'+input_key
    upper_df=df[volt_name].between(4.1,4.2)
    lower_df=df[volt_name].between(3.9,4.05)
    max1 = upper_df[out_name].max()
    max2 = lower_df[out_name].max()
    diff = max1 - max2
    return(diff)

def derivative(df, df_x, df_y, input_key):
    dy = np.zeros((len(df_x)))
#    dy[0:-1] = (np.diff(df_y)/np.diff(df_x))
#    dy[-1] = 'NaN'
    dy = np.gradient(df_y,df_x,edge_order=2)
    long_key = 'd' + input_key
    df[long_key] = dy
    return df[long_key]

def Log_Mfact(Mprime):
    return(log(factorial(Mprime)))

def logOmega(j, N, Mprime, log_Mfact, overlit):
    # Can I convert to a Stirling approximation?
    if N < Mprime:
#        print '1 =', M - N + j, ' ;2 =', N - j, ' ;3 =', M - j
           result_reduced = log_Mfact - fact_array[Mprime - N + j] - fact_array[N - j] - fact_array[Mprime - j] - fact_array[j]

    else:
#        print '1 =', 2*M - N - j, ' ;2 =', N - M + j, ' ;3 =', M - j
           result_reduced = log_Mfact - fact_array[2*Mprime - N - j] - fact_array[N - Mprime + j] - fact_array[Mprime - j] - fact_array[j]
    result = result_reduced + log_Mfact
    return result 

def E(j, N, J1, J2, deltaE, Edel,overlit,Mprime):
    # Now in meV!
    M = int(Mprime / (1 - 3 * overlit/100))
    E1 = Edel / 2
    E2 = - Edel / 2
    x = overlit/100

    if N < Mprime:
        z1 = (N - j) / Mprime
        z2 = j / Mprime
    else:
        z1 = (Mprime - j) / Mprime
        z2 = (N - Mprime + j) / Mprime
    sub1occ = (z1 * (1 - 3 * x)) + 3 * x
    sub2occ = (z2 * (1 - 3 * x)) + 3 * x
    E1star = (E1 + (0.5 * (J2 + deltaE) * Nsecond) * sub1occ + (0.5 * J1 * Nfirst) * sub2occ) * e * 0.001
    E2star = (E2 + (0.5 * (J2 - deltaE) * Nsecond) * sub2occ + (0.5 * J1 * Nfirst) * sub1occ) * e * 0.001
    result = ((sub1occ * M * E1star) + (sub2occ * M * E2star))
    result = np.longdouble(result)
    return result 

def logSumExp(ns):
    max = np.max(ns)
    ds = ns - max
    sumOfExp = np.exp(ds).sum()
    return max + np.log(sumOfExp)

def logQ(N, J1, J2, energy_diff, Edel, overlit,Mprime,T):
    beta = 1/(k_B * T)
    log_Mfact=Log_Mfact(Mprime)
    q_result = 0
    if N < Mprime:
        logq_result = logSumExp(np.array([logOmega(j, N, Mprime, log_Mfact, overlit) + (-E(j, N, J1, J2, energy_diff,Edel,overlit,Mprime) * beta) for j in range(0,N + 1)]))
#            print exp(-E(j,N,deltaE)*beta), 'exp'
    else:
        logq_result = logSumExp(np.array([logOmega(j, N, Mprime, log_Mfact, overlit) + (-E(j, N, J1, J2, energy_diff,Edel,overlit,Mprime) * beta) for j in range(0, 2*Mprime - N + 1)]))
    result=(np.longdouble(logq_result))
 #   print Omega(j,N)
    return result

def integrate(df,df_x, df_y, input_key):
    int_key = str(input_key).lstrip('d')
    d_x = float(df_x.iloc[1] - df_x.iloc[0])
    df[int_key] = 0.0
    for j in range(1,len(df_y)):
        y_array=np.array(df_y[1:j+1])
#        print d_x, 'dx'
        df[int_key][j] = np.trapz(y = y_array, dx = d_x)
#        print df[int_key][j], 'int_key'
#        print df[int_key][j]
    return(df[int_key])

def S(N, deltaE, J1,J2, logQ_array,Edel,overlit,logMfact,Mprime,T):
    beta = 1/(k_B * T)
    if N == 0 or N == 2 * Mprime:
        return(0)
    else:
        omega_sum= 0
        if N < Mprime:
            beta_array = np.array([beta*E(j, N, J1, J2, deltaE,Edel,overlit,Mprime) for j in range(0 , N + 1)])
#            print 'log beta_array=', np.log(beta_array)
#            print 'beta_array=', beta_array
            omega_sum = np.sum([beta_array[j] * np.exp(-beta_array[j] + logOmega(j , N, Mprime,logMfact,overlit) - logQ_array[N]) for j in range(0 , N + 1)])
#                print 'Omega = ', Omega(j, N), ' ;E =  ', E(j , N)
        elif N >= Mprime and N < 2 * Mprime:
            beta_array = np.array([beta*E(j, N, J1, J2, deltaE,Edel,overlit,Mprime) for j in range(0 , 2 * Mprime - N + 1)])
            omega_sum = np.sum([beta_array[j] * np.exp(-beta_array[j] + logOmega(j , N, Mprime,logMfact,overlit) - logQ_array[N]) for j in range(0 , 2 * Mprime - N + 1)])
#        print 'prob', '{:.2e}'.format(np.exp(-E(j, N) * beta))
 #       print 'omega_sum=', omega_sum, 'logQ_array=', logQ_array[N] 
        result=k_B * (omega_sum + logQ_array[N]) # Shifted Q_array index.
        result=np.longdouble(result)
#        print 'result=', '{:.2e}'.format(result), ' ;N=' ,N
        return(result)

def mu(df_x,df_logQ,energy_diff,dlogQ_array,E0,T):
    # Is in eV!
    mu_result = - (k_B * T * dlogQ_array + epsilon * e) / e - E0
    return(mu_result)

def dxdmu(df,df_x,df_mu,dlogQ_array,energy_diff,long_dxdmu,M,T):
    N_max = 2 * M   
    result = (- 1 / (k_B * T * derivative(df, df_x, dlogQ_array, long_dxdmu))) * e / N_max # Units: per joule. Normalised per particle.
    return(result)

def long_var(var,energy_diff):
    return(str(var + '_%.2f' % energy_diff))

def long_var_dict(long_dict,var,energy_diff):
    long_dict.setdefault(var, [])
    long_dict[var].append([long_var(var,energy_diff), '{0:.2f}'.format(energy_diff)])

def peak_height_anal(df_y):
    size_of_df = len(df_y)
    lower_slice = int(size_of_df * 0.35)
    upper_slice = int(size_of_df * 0.65)
    reduced_df = df_y[lower_slice:upper_slice]
    lower_amp = min(reduced_df)
    upper_amp = max(reduced_df)
    total_amp = upper_amp - lower_amp
    return (lower_amp,upper_amp,total_amp)

def subocc(df_x,df_H, E0, J1, J2, T, delta,M):
    # Only works for delta = 0 at the moment!
    # Takes the enthalpy per mole and converts to absolute enthalpy.
    J1prime = J1 * k_B * T / e
    J2prime = J2 * k_B * T / e
    E0prime = -E0
    df_Habs = (df_H * 1000 / R) * k_B / e
#    print 'J1=', J1prime, 'J2=', J2prime, 'E0=', E0prime
    a = 4 * J1prime - 12 * J2prime
#    print 'a=', a
    if a == 0:
        n1 = df_x
        n2 = df_x
    else:
        b = 2 * df_Habs - 4 * E0prime * df_x - 24 * J2prime * (df_x * df_x)
        C = b / a
        B = - 2 * df_x
        A = 1
        n2 = (-B - np.sqrt(B * B - 4 * A * C)) / (2 * A)
        n1 = 2 * df_x - n2
#    n2 = 1/(2*J1*Nfirst) * (df_Habs - df_x * (2 * J1 * Nfirst + J2 * Nsecond))
#    n1 = df_x - (2* df_Habs)
#    else:
#        n2 = df_x
#        n1 = df_x
 
    return([n1,n2])

# df=pd.DataFrame()
# Calculate all parameters with g=0 as input.
def var_check(key,full_list,long_var,Ediff):
    if key == long_var:
        return(Ediff)
    else:
        value = full_list[key][0]
        return(value)

def g_zero_solution(arg_dict,counter):
    for variable in arg_dict:
        if not hasattr(arg_dict[variable],"__iter__") and variable not in ('label','Mprime'):
            arg_dict[variable]=[arg_dict[variable]]
    E0 = arg_dict['E0']
    deltaE = arg_dict['deltaE']
    J1 = arg_dict['J1']
    J2 =arg_dict['J2']
    delta = arg_dict['delta']
    overlit = arg_dict['overlit']
    T = arg_dict['T']
    Svib = arg_dict['Svib']
    Mprime = arg_dict['Mprime']
    lab=arg_dict['label']

    dataframe_dict = {}
#    print arg_dict
    for variables in zip(E0,deltaE,J1,J2,delta,overlit,T,Svib):
        print('variables =', variables)
        df=pd.DataFrame()
        counter+=1
#       print variables, 'variables'
        E0 = variables[0]
        deltaE =variables[1]
        J1 = variables[2]
        J2 = variables[3]
        delta = variables[4]
        overlit = variables[5]
        T = variables[6]
        Svib = variables[7]
        beta = 1/(k_B * T)
#        print 'Ediff=', Ediff, type(Ediff)
        energy_diff = arg_dict[lab][counter-1]
        Ediff = energy_diff
#        g_value = Ediff
        long_S_gzero = long_var('S', energy_diff)
#    print S(2, energy_diff), 'example_S_value'
#    df['S'] = np.array([S(i, energy_diff) / (2*M*k_B) for i in range(0, N_max)]) # in format plotted in paper.
        y = overlit
        xpinned = 3 * y / 100
        print('Mprime =', Mprime)
        print('y =', y)
        M = int(Mprime/(1 - 3 * y/100))
        print('M = ', M)
        
        N_max = 2 * M
        N_array = np.array([i for i in range(0,2*Mprime)])

        df['N'] = N_array
        df['x'] = xpinned + N_array / N_max
        df['xmobile'] = N_array / (2 * Mprime)

        long_logQ = long_var('logQ', energy_diff)
        long_mu_gzero = long_var('mu', energy_diff)
        long_dxdmu_gzero = long_var('dxdmu', energy_diff)
        long_dxmobdmu_gzero = long_var('dxmobdmu',energy_diff)
     
#        print 'y=', y,'M=', M, 'M_prime=', Mprime
        logQ_array = (np.array([logQ(N, J1, J2, delta, deltaE,overlit,Mprime,T) for N in range(0,2*Mprime)])) 
#    print Q_array
#        logQ_array = np.log(Q_array , dtype=np.longdouble) # Performed elementwise
        df['logQ'] = logQ_array

        dlogQ_array = derivative(df, df['N'], df['logQ'], long_logQ)
        long_V_gzero = long_var('V', energy_diff)

        df[long_mu_gzero] = mu(df['N'], df['logQ'], delta, dlogQ_array,E0,T)
        df[long_dxdmu_gzero] = dxdmu(df,df['N'], df[long_mu_gzero], dlogQ_array, delta, long_dxdmu_gzero, M,T)


        df[long_dxmobdmu_gzero] = df[long_dxdmu_gzero] * derivative(df, df['x'], df['xmobile'], 'xmobile')  
        df[long_V_gzero] = - df[long_mu_gzero]
#    long_Salt=long_var('Salt',energy_diff)
        long_total_amp = long_var('total_amp', energy_diff)
        long_min_amp = long_var('min_amp', energy_diff)
        long_max_amp = long_var('max_amp', energy_diff)

#    dS = np.zeros((len(df['x'])))
#    dS[0:-1] = (np.diff(df[long_S])/np.diff(df['x']))
#    dS[-1] = (df[long_S].iloc[-1] - df[long_S].iloc[-2]) / (df['x'].iloc[-1] - df['x'].iloc[-2])

#    print type(df['S'][0])
#    print df[long_S]

        long_U_gzero = long_var('U',energy_diff)
        long_dU_gzero = long_var('dU',energy_diff)
        long_dS_gzero = long_var('dS',energy_diff)
        long_dSmob_gzero = long_var('dSmob',energy_diff)
        long_dSmobvib_gzero = long_var('dSmobvib',energy_diff)
        long_dH_gzero = long_var('dH',energy_diff)
        long_dG_gzero = long_var('dG',energy_diff)
        long_x_gzero = long_var('x',energy_diff)
        long_xmob_gzero = long_var('xmobile',energy_diff)

        df[long_x_gzero] = df['x']
        df[long_xmob_gzero] = df['xmobile']
        logMfact = Log_Mfact(Mprime)
        df[long_S_gzero] = np.array([S(i, delta,J1,J2,logQ_array,deltaE,overlit,logMfact,Mprime,T) * R / (2*Mprime*k_B) for i in range(0, 2*Mprime)]) # in format plotted in paper.
        df[long_dS_gzero] = derivative(df,df['x'],df[long_S_gzero],long_S_gzero)
        df[long_dSmob_gzero] = df[long_dS_gzero] * derivative(df, df['xmobile'], df['x'], 'x')
        df[long_dSmobvib_gzero] = df[long_dSmob_gzero] + Svib # Vibrational correction.
        df[long_dH_gzero] = (df[long_mu_gzero] * e + T * df[long_dS_gzero] * k_B / R) * R / (k_B *1000)

        df[long_dG_gzero] = df[long_dH_gzero] - (T * df[long_dS_gzero] / 1000)
#    df[long_Salt] = integral(df['x'],df[long_dS])
        long_H_gzero = long_var('H' , energy_diff)
        long_G_gzero = long_var('G', energy_diff)

#        df[long_H_gzero] = integrate(df,df['x'], df[long_dH_gzero],long_dH_gzero)
#        df[long_G_gzero] = integrate(df,df['x'], df[long_dG_gzero],long_dG_gzero)

#        df[long_dH_gzero] = derivative(df['x'], df[long_H_gzero],long_H_gzero)
#        df[long_dG_gzero] = derivative(df['x'], df[long_G_gzero],long_G_gzero)

#        min_dH = df[long_dH_gzero][1:-2].min()
#        max_dH = df[long_dH_gzero][1:-2].max()
#        print min_dH

#        long_n1 = long_var('n1', energy_diff)
#        long_n2 = long_var('n2', energy_diff)
#        df[long_n1] = subocc(df['x'],df[long_H_gzero],E0,J1,J2,delta,M,T)[0]
#        df[long_n2] = subocc(df['x'],df[long_H_gzero],E0,J1,J2,delta,M,T)[1]
#        print 'dH', df[long_H_gzero][0:10]
#        print 'dG', df[long_G_gzero][0:10]

        df[long_min_amp] = peak_height_anal(df[long_dSmob_gzero])[0]
        df[long_max_amp] = peak_height_anal(df[long_dSmob_gzero])[1]
        df[long_total_amp] = peak_height_anal(df[long_dSmob_gzero])[2]
#        print 'Min amp=', df[long_min_amp][0]
#        print 'Max_amp=', df[long_max_amp][0]
        print('Overall Entropy Amplitude=', df[long_total_amp][0])
        for var in var_list:
            long_var_dict(long_dict,var,energy_diff)

#        df.to_csv('output/all_data_'+'multi_val' + '=' + str(Ediff)+ '.csv')
        dataframe_dict['%.2f' % energy_diff]=df
        print('*********End of iteration*********\n\n')
    return(dataframe_dict)

def plotter(df_dict, long_dict,loc_value):
    for key,value_list in long_dict.items():
        print('key,value_list', key, value_list)
        if key == 'n1':
            for value in value_list:
                suffix = value[0].split('_')[-1]
                plt.plot(df_dict[value[1]]['x'],df_dict[value[1]][str(value[0])], label=str(value[1]) + ', n1')
                plt.plot(df_dict[value[1]]['x'],df_dict[value[1]]['n2_' + suffix], label=str(value[1]) + ', n2')
        elif key != 'n2':
            for value in value_list:
#                plt.plot(df_dict[value[1]]['x'],df_dict[value[1]][str(value[0])], label=str(value[1]))
                plt.plot(df_dict[value[1]]['xmobile'],df_dict[value[1]][str(value[0])], label=str(value[1]))
        plt.xlabel('$x_{r}$')
        plt.ylabel(str(var_names[key]) + ' / ' + str(units[key]))
        if key != 'S':
            plt.legend(loc=loc_value,fontsize=16)
#        plt.ylim([0,8])    
        plt.savefig('output/%svsx.png'% str(key),dpi=300)
        plt.clf()
        if key == 'n1':
            for value in value_list:
                suffix = value[0].split('_')[-1]
                plt.plot(df_dict[value[1]]['n1_' + suffix], df_dict[value[1]]['H_' + suffix], label=str(value[1]) + ', n1')
                plt.plot(df_dict[value[1]]['n2_' + suffix], df_dict[value[1]]['H_' + suffix], label=str(value[1]) + ', n2')
        plt.xlabel('H / kJ mol-1')
        plt.ylabel(str(var_names[key]) + ' / ' + str(units[key]))
        if key != 'S':
            plt.legend(loc=loc_value,fontsize=16)
        plt.savefig('output/%svsH.png'% str(key),dpi=300)
        plt.clf()


    voltages=long_dict['V']
    voltage_list=[]
    for entry in voltages:
        voltage_list.append(entry[0])
    print(voltage_list)

    for key,value_list in long_dict.items():    
        for n,value in enumerate(value_list):
            plt.plot(df_dict[value[1]][voltage_list[n]][1:-2],df_dict[value[1]][str(value[0])][1:-2], label=str(value[1]))
        plt.xlabel('E / V vs. Li')
        plt.ylabel(str(var_names[key]) + ' / ' + str(units[key]))
        plt.legend(loc=loc_value,fontsize=16)
        plt.ylim([0,8])
        plt.savefig('output/%svsV.png'% str(key),dpi=300)
        plt.clf()

def alt_plotter(df_dict,long_dict,loc):
    for key, df in df_dict.items():
        plt.plot(df['V_'+str(key)],df['dxdmu_'+str(key)],label=key)
    plt.legend(loc=0)
    plt.show()
    print(list(df_dict))

def sub_plotter(df_dict,long_dict):
    f, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2, sharex='col', figsize =(7,7))
    axes=(ax1,ax3,ax2,ax4)
    print(list(long_dict.keys()))
    for key, df in sorted(df_dict.items()):
        ax1.plot(df['x'],df['V_' + str(key)],label='$T= $'+'%.3g' % float(key))
        ax3.plot(df['x'],df['dxdmu_'+str(key)],label='$T= $'+ '%.3g' % float(key))
        ax2.plot(df['x'],df['S_'+str(key)],label='$T= $'+'%.3g' % float(key))
        ax4.plot(df['x'],df['dS_'+str(key)],label='$T= $'+ '%.3g' % float(key))
    ax1.set_ylabel('Voltage vs. Li/Li$^{+}$ (V)')
    ax3.set_ylabel('d$x$/d$V$ (V$^{-1}$)')
    ax2.set_ylabel('S (J mol$^{-1}$ K$^{-1}$)')
    ax1.get_yaxis().set_label_coords(-0.18,0.5)
    ax2.get_yaxis().set_label_coords(1.22,0.5)
    ax3.get_yaxis().set_label_coords(-0.18,0.5)
    ax4.get_yaxis().set_label_coords(1.22,0.5)
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax4.yaxis.tick_right()
    ax1.xaxis.tick_top()
    ax2.xaxis.tick_top()
    ax2.yaxis.set_ticks_position('both')
    ax4.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax2.xaxis.set_ticks_position('both')
    ax4.yaxis.set_label_position('right')
    ax4.set_ylabel('d$S$/d$x$ (J mol$^{-1}$ K$^{-1}$)')
    ax4.set_ylim([-35,38])
    ax2.set_ylim([-0.5,6.5])
    ax1.set_yticks(np.arange(3.7,4.3,0.1))
    ax1.set_xlabel('Total 8a Li content, $x$')
    ax1.xaxis.set_label_position('top')
    ax2.xaxis.set_label_position('top')
    ax2.set_xlabel('Total 8a Li content, $x$')
    ax1.legend(loc=3,fontsize=10.5,handletextpad=0.1)
    for n,ax in enumerate(axes):
        if ax == ax1 or ax == ax3:
            ax.text(0.83,0.87,'('+string.ascii_lowercase[n]+')',transform=ax.transAxes,size=15,weight='demi')
        else:
            ax.text(0.05,0.87,'('+string.ascii_lowercase[n]+')',transform=ax.transAxes,size=15,weight='demi')
#    ax1.set_ylim([3.65,4.25])
    ax3.set_ylim([-0.5,10.5])
    plt.tight_layout()
    f.subplots_adjust(hspace=0)
    plt.savefig('output/modelfig1.png',dpi=400)
    plt.show()
def getkey(dict_iteritems):
    return(float(dict_iteritems[0]))

def column_plot(df_dict,long_dict):
    f, ((ax1, ax2,ax3)) = plt.subplots(3,1, figsize=(4.5,9),sharex='col')
    axes = (ax1, ax2, ax3)
    print(list(long_dict.keys()))
    for k, df in sorted(iter(df_dict.items()),key=getkey):
        lab = '$T$ = %d' % (float(k) - 273)
#        lab = '$J_{2}$ = ' +k
        ax1.plot(df['V_' + str(k)],df['xmobile'],label=lab)
        ax2.plot(df['V_' + str(k)],df['dxmobdmu_'+str(k)],label=lab)
        ax3.plot(df['V_' + str(k)],df['dSmob_'+str(k)],label=lab)
    ax1.set_ylabel('Removable Li, $x_{r}$')
    ax2.set_ylabel('d$x_{r}$/d$V$ (V$^{-1}$)')
    ax3.set_ylabel('d$S$/d$x_{r}$ (J mol$^{-1}$ K$^{-1}$)')
    ax1.get_yaxis().set_label_coords(-0.1,0.5)
    ax2.get_yaxis().set_label_coords(-0.1,0.5)
    ax3.get_yaxis().set_label_coords(-0.1,0.5)
    ax1.yaxis.set_label_position('right')
    ax1.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax3.yaxis.set_label_position('right')
    ax3.yaxis.tick_right()
    ax1.set_ylim([-0.05,1.05])
    ax2.set_ylim([-0.05,11.3])
    ax3.set_ylim([-25,38])
    for n,ax in enumerate(axes):
        if ax == ax1:
            ax.text(0.03,0.1,'('+string.ascii_lowercase[n]+')',transform=ax.transAxes,size=15,weight='demi')
        else:
            ax.text(0.03,0.87,'('+string.ascii_lowercase[n]+')',transform=ax.transAxes,size=15,weight='demi')
#    ax3.set_xticks(np.arange(3.8,4.3,0.0.05))
    ax3.set_xlabel('Voltage vs. Li/Li$^{+}$ (V)')

    ax1.legend(loc=0,fontsize=11)
    plt.tight_layout()
    f.subplots_adjust(hspace=0)
    plt.savefig('output/column_fig.png',dpi=400)
    plt.show()

def sub_plotter_voltage(df_dict,long_dict):
    f, ((ax1, ax2)) = plt.subplots(1,2, sharex='col', figsize =(7,4))
    axes=(ax1,ax2)
    print(list(long_dict.keys()))
    for key, df in sorted(df_dict.items()):
        ax1.plot(df['V_' + str(key)],df['dxdmu_'+str(key)],label='$J_{1}=$'+'%.3g' % float(key))
        ax2.plot(df['V_' + str(key)],df['dS_'+str(key)],label='$J_{1}=$'+ '%.3g' % float(key))
#    ax1.get_yaxis().set_label_coords(-0.5,20.5)
#    ax2.get_yaxis().set_label_coords(-35,38)
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax1.set_xticks(np.arange(3.8,4.3,0.1))
    ax2.set_xticks(np.arange(3.8,4.3,0.1))
    ax1.legend(loc=0,fontsize=10.5,handletextpad=0.1)
    ax1.text(0.83,0.87,'(a)',transform=ax1.transAxes,size=15,weight='demi')
    ax2.text(0.08,0.87,'(b)',transform=ax2.transAxes,size=15,weight='demi')
    for axis in axes:
        axis.xaxis.set_ticks_position('both')
        axis.yaxis.set_ticks_position('both')
    ax1.set_xlim([3.75,4.25])
    ax2.set_xlim([3.75,4.25])
    ax1.set_ylim([-0.05,12.5])
    ax2.set_ylim([-35,38])
    ax1.set_xlabel('Voltage vs. Li/Li$^{+}$ (V)')
    ax2.set_xlabel('Voltage vs. Li/Li$^{+}$ (V)')
    ax1.set_ylabel('d$x$/d$V$ (V$^{-1}$)')
    ax2.set_ylabel('d$S$/d$x$ (J mol$^{-1}$ K$^{-1}$)')
    plt.tight_layout()
    plt.savefig('output/modelfig1_voltage.png',dpi=400)
    plt.show()

def sub_plotter_rem(df_dict,long_dict):
    f, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2, sharex='col', figsize =(7,6))
    axes=(ax1,ax3,ax2,ax4)
    print(list(long_dict.keys()))
    for key, df in sorted(df_dict.items()):
        ax1.plot(df['xmobile'],df['V_' + str(key)],label='$J_{1}=$'+key)
        ax3.plot(df['xmobile'],df['S_' +str(key)],label='$J_{1}=$'+key)
        ax2.plot(df['V_' + str(key)],df['dxmobdmu_'+str(key)],label='$J_{1}=$'+key)
        ax4.plot(df['V_' + str(key)],df['dSmob_'+str(key)],label='$J_{1}=$'+key)
    ax1.set_ylabel('Voltage vs. Li/Li$^{+}$ (V)')
    ax3.set_ylabel('S (J mol$^{-1}$ K$^{-1}$)')
    ax2.set_ylabel('d$x_{r}$/d$V$ (V$^{-1}$)')
    ax1.get_yaxis().set_label_coords(-0.18,0.5)
    ax2.get_yaxis().set_label_coords(1.22,0.5)
    ax3.get_yaxis().set_label_coords(-0.18,0.5)
    ax4.get_yaxis().set_label_coords(1.22,0.5)
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax4.yaxis.tick_right()
    for axis in axes:
        axis.xaxis.set_ticks_position('both')
        axis.yaxis.set_ticks_position('both')
    ax4.yaxis.set_label_position('right')
    ax4.set_ylabel('d$S$/d$x_{r}$ (J mol$^{-1}$ K$^{-1}$)')
    ax4.set_xticks(np.arange(3.8,4.3,0.1))
    ax4.set_ylim([-35,38])
    ax2.set_ylim([-0.5,10.5])
    ax1.set_yticks(np.arange(3.8,4.3,0.1))
    ax3.set_xlabel('Removable Li content, $x_{r}$$')
    ax4.set_xlabel('Voltage vs. Li/Li$^{+}$ (V)')
    ax1.legend(loc=0,fontsize=10.5,handletextpad=0.1)
    for n,ax in enumerate(axes):
        if ax == ax1 or ax == ax3:
            ax.text(0.83,0.87,'('+string.ascii_lowercase[n]+')',transform=ax.transAxes,size=15,weight='demi')
        else:
            ax.text(0.05,0.87,'('+string.ascii_lowercase[n]+')',transform=ax.transAxes,size=15,weight='demi')
    ax1.set_ylim([3.75,4.25])
    ax3.set_ylim([-0.5,6.5])
#    plt.tight_layout()
    f.subplots_adjust(hspace=0)
    plt.savefig('output/modelfig1.png',dpi=400)
    plt.show()

def double_column_plot(df_dict,long_dict):
    f, ((ax1, ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2, figsize=(7,8), sharex='col')
    axes=(ax1,ax2,ax3,ax4,ax5,ax6)
    print(list(long_dict.keys()))
    for k, df in sorted(iter(df_dict.items()),key=getkey):
        overlit_frac = float(k)
        lab = '$y = $' + '%.1f' % overlit_frac
        ax1.plot(df['V_' + str(k)],df['x'],label=lab)
        ax3.plot(df['V_' + str(k)],df['dxdmu_'+str(k)],label=lab)
        ax5.plot(df['V_' + str(k)],df['dS_'+str(k)],label=lab)
        ax2.plot(df['V_' + str(k)],df['xmobile'],label=lab)
        ax4.plot(df['V_' + str(k)],df['dxmobdmu_'+str(k)],label=lab)
        ax6.plot(df['V_' + str(k)],df['dSmob_'+str(k)],label=lab)
    ax1.set_ylabel('Total 8a Li, $x$')
    ax3.set_ylabel('d$x$/d$V$ (V$^{-1}$)')
    ax5.set_ylabel('d$S$/d$x$ (J mol$^{-1}$ K$^{-1}$)')
    ax2.set_ylabel('Removable Li, $x_{r}$')
    ax4.set_ylabel('d$x_{r}$/d$V$ (V$^{-1}$)')
    ax6.set_ylabel('d$S$/d$x_{r}$ (J mol$^{-1}$ K$^{-1}$)')
    ax1.get_yaxis().set_label_coords(-0.15,0.5)
    ax2.get_yaxis().set_label_coords(1.18,0.5)
    ax3.get_yaxis().set_label_coords(-0.15,0.5)
    ax4.get_yaxis().set_label_coords(1.18,0.5)
    ax5.get_yaxis().set_label_coords(-0.15,0.5)
    ax6.get_yaxis().set_label_coords(1.18,0.5)
    for axis in axes:
        axis.xaxis.set_ticks_position('both')
        axis.yaxis.set_ticks_position('both')
        axis.set_xticks(np.arange(3.9,4.3,0.1))
    ax1.set_ylim([-0.05,1.05])
    ax3.set_ylim([-0.05,11])
    ax5.set_ylim([-25,38])
    ax2.set_ylim([-0.05,1.05])
    ax4.set_ylim([-0.05,11])
    ax6.set_ylim([-25,38])
    ax2.yaxis.set_label_position('right')
    ax4.yaxis.set_label_position('right')
    ax6.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax4.yaxis.tick_right()
    ax6.yaxis.tick_right()
    ax2.yaxis.set_ticks_position('both')
    ax4.yaxis.set_ticks_position('both')
    ax6.yaxis.set_ticks_position('both')
#    ax3.set_xticks(np.arange(3.8,4.3,0.0.05))
    ax5.set_xlabel('Voltage vs. Li/Li$^{+}$ (V)')
    ax6.set_xlabel('Voltage vs. Li/Li$^{+}$ (V)')
    ax2.legend(loc=0,fontsize=10.5,handletextpad=0.1)
    for n,ax in enumerate(axes):
        if ax == ax1 or ax == ax2:
            ax.text(0.05,0.1,'('+string.ascii_lowercase[n]+')',transform=ax.transAxes,size=15,weight='demi')
        else:
            ax.text(0.05,0.87,'('+string.ascii_lowercase[n]+')',transform=ax.transAxes,size=15,weight='demi')
    plt.tight_layout()
    f.subplots_adjust(hspace=0)
    plt.savefig('output/double_column_fig.png',dpi=400)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Put in some energy parameters')
    parser.add_argument('--E0', type=float, help='Adds a point term', default = [0.0], nargs='+')
    parser.add_argument('--Svib', type=float, help = 'Vibrational correction', default= [0.0], nargs='+')
    parser.add_argument('--deltaE', type=float, help='Separation between point terms.', default = [0.0], nargs='+')
    parser.add_argument('--J1', type=float, help='Nearest neighbour parameter.', default = [0.0], nargs='+')
    parser.add_argument('--J2', type=float, help='Next nearest neighbour parameter.', default = [0.0], nargs='+')
    parser.add_argument('--delta', type=float, help='Separation next nearest neighbours on each sublattice.', default = [0.0], nargs='+')
    parser.add_argument('--loc', type=int, help='Legend_location', default = 0)
    parser.add_argument('--overlit',type=int, help = 'Overlithiation values, as percentage', default = [0], nargs='+')
    parser.add_argument('--Mprime',type=int, help = 'removable particles in each sublattice', default = 100)
    parser.add_argument('--T',type=float, help = 'particles in each sublattice', default = [293.0], nargs='+')
    parser.add_argument('--nargs',type= int, help = 'number of arguments', default = 3)
    #parser.add_argument
    parser.add_argument('--label', type=str,help= 'variable to be labelled in plots.', default = 'overlit')
 
    args = parser.parse_args(argv[1:])
    
    E0 = args.E0
    deltaE = args.deltaE
    J1 = args.J1
    J2 = args.J2
    T = args.T
    delta = args.delta
    overlit = args.overlit
    Mprime = args.Mprime
    Svib = args.Svib

    arg_dict = dict(vars(args))
    del arg_dict['loc']
    del arg_dict['nargs']

    for key,value in arg_dict.items():
        if key != 'Mprime' and key != 'label':
            while len(value) < args.nargs:
                arg_dict[key].append(value[-1])
    
    print('Full list: ', arg_dict)
            
    print('E0 = ', E0)
    print('deltaE = ', deltaE)
    
    counter = 0
    df_dict = g_zero_solution(arg_dict,counter)
#    plotter(df_dict, long_dict, args.loc)
#    double_column_plot(df_dict,long_dict)
    sub_plotter(df_dict,long_dict)
#    column_plot(df_dict,long_dict)
 
