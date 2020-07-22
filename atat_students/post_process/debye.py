# Plotting of different hypotheses based on the paper of
# Reynier, Yazami, Fultz
# Paramters from Zaghib et al: "Effective and Debye temperatures of alkali-metal atoms in graphite intercalation compounds

# Looks like carbon vibrations are neglected.
# Parallel vibration modes theta_par

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import os
import string
from scipy import integrate

# plt.style.use('classic')
# plt.figure(100)
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

mpl.rcParams['lines.linewidth'] = 2.5

font = {'size': 20}

mpl.rc('font', **font)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams.update({'errorbar.capsize': 2})
mpl.rcParams['legend.handlelength'] = 0



k_b = 1.38e-23 # Boltzmann constant, SI units
R = 8.92 # Molar gas constant, J mol-1 K-1
theta_bccli = 380 # Debye temperature for metallic bcc Li, in K
theta_parr_SI = 392 # Debye T from Zaghib for Li vibrating parallel. Could not determine difference
# theta_parr_SII = 392
theta_parr_SII = 330
theta_parr_SIID = 320
theta_perp = 893 # Debye temperature of perpendicular vibration

def delta_Svib_fultz(x):
    c_stagei = (x - 0.5) *2
    c_stageii = (1 - c_stagei)
    theta_parr = theta_parr_SI *x
    term_1 = R * np.log(theta_bccli/theta_perp)
    term_2 = 2 * R * np.log(theta_bccli/theta_parr)
    print('perp_term=', term_1)
    print('parr_term=', term_2)
    return(term_1 + term_2)

def delta_Svib_mercer(x):
    size = x.size
    print('size=',size)
    outarray = np.empty(size)
    print(outarray)
    for i,xval in enumerate(x):
        if(xval >= 0.5):
            c_stagei = (xval - 0.5) *2
            c_stageii = (1 - c_stagei)
#            print('C-stagei',c_stagei)
#            print('C-stageii',c_stageii)    
            theta_parr = c_stagei * theta_parr_SI + c_stageii * theta_parr_SII
            term_1 = R * np.log(theta_bccli/theta_perp)
            term_2 = 2 * R * np.log(theta_bccli/theta_parr)
#            print('perp_term=', term_1)
#            print('parr_term=', term_2)
            outarray[i] = term_1 + term_2
        elif(xval < 0.5):
            c_stageii = (xval - 0.3333333) * 6
            c_stageiid = 1 - (xval - 0.3333333) * 6
            print('C-stageii',c_stageii)
            print('C-stageiid',c_stageiid)
            theta_parr = c_stageii * theta_parr_SII + c_stageiid * theta_parr_SIID
            term_1 = R * np.log(theta_bccli/theta_perp)
            term_2 = 2 * R * np.log(theta_bccli/theta_parr)
#            print('perp_term=', term_1)
#            print('parr_term=', term_2)
            outarray[i] = term_1 + term_2            
    return(outarray)        

x = np.linspace(0,1,68)
ones = np.ones(68)
dSvib_fultz = delta_Svib_fultz(ones)
dSvib_mercer = delta_Svib_mercer(x)
T_K = 320 # temeprature in kelvin
F = 96485 # Faraday constant

file_list = os.listdir('.')
entropy_file = [f for f in file_list if f.endswith('entropy.csv')][0]
basytec_file = [f for f in file_list if f.endswith('txt')][0]

df_e = pd.read_csv(entropy_file,encoding='latin')
df_b = pd.read_csv(basytec_file,skiprows=12,encoding='latin')

df_e['SOC'] = df_e['Charge/Discharge [mAh]']/df_e['Charge/Discharge [mAh]'].iloc[-1]

df_e['M3 Enthalpy [J mol-1]'] = - F * df_e['OCV [V]   '] + T_K * df_e['M3 Entropy [J mol-1 K-1]']

interp_x = df_e['SOC'].values
df_e['Entropy_corrected'] = df_e['M3 Entropy [J mol-1 K-1]'] - delta_Svib_mercer(interp_x)
df_e['Entropy_pcorrected'] = df_e['M3 Entropy [J mol-1 K-1]'] - delta_Svib_fultz(ones)

plt.plot(df_e['SOC'][1:-2], df_e['M3 Entropy [J mol-1 K-1]'][1:-2],label='Raw data',marker='o')
plt.plot(df_e['SOC'][1:-2], df_e['Entropy_corrected'][1:-2],label='Corrected for Svib (Schirmer)',marker='^')
plt.plot(x,dSvib_fultz,label='Reynier correction',linestyle='--',marker='>',markersize=3)
plt.plot(x,dSvib_mercer,label='Reynier correction, Schirmer parameters',linestyle='--',marker='<',markersize=3)
plt.legend(fontsize=12)
plt.xlabel('Lithium fraction x')
plt.ylabel('dS/dx (J mol$^{-1}$ K$^{-1}$)')
plt.tight_layout()
plt.savefig('plot_320K.png')
plt.show()

plt.clf()

df_e['S_config'] = integrate.cumtrapz(y = df_e['Entropy_corrected'].values,x = df_e['SOC'].values,initial=0)
# df_e['S_config_partial'] = integrate.cumtrapz(y = df_e['Entropy_pcorrected'].values,x = df_e['SOC'].values,initial=0)
# df_e['S_config_raw'] = integrate.cumtrapz(y = df_e['M3 Entropy [J mol-1 K-1]'].values,x = df_e['SOC'].values,initial=0)
df_e['S_ss']=-R * (df_e['SOC']*np.log(df_e['SOC']) + (1-df_e['SOC'])*np.log(1-df_e['SOC']))
plt.plot(df_e['SOC'].iloc[1:-2]*0.9,df_e['S_config'].iloc[1:-2],label='corrected')
df_e.to_csv('lithiation_proc.csv')
print(df_e)
# plt.plot(df_e['SOC']*0.9,df_e['S_config_partial'],label='partial correction')
# plt.plot(df_e['SOC']*0.9,df_e['S_config_raw'],label='raw')
plt.plot(df_e['SOC'],df_e['S_ss'],label='ss')
plt.legend()
plt.show()
           
  
