import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import string
from dqdv_proc import DQDV
from scipy.integrate import cumtrapz

# Directories.

entropy_path = '../entropy_data/'
figures_path = '../figures/'
galvan_path = '../galvanostatic/'

# Figure formatting.

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

mpl.rcParams['lines.linewidth'] = 2.5

font = {'size': 18}

mpl.rc('font', **font)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams.update({'errorbar.capsize': 2})
mpl.rcParams['legend.handlelength'] = 2
temp_dirs = [entropy_path +'entropydischarge_hc_00/']
T_list = ['dis','dis','dis','dis','dis']
method = ['M1','M2','M3','M4','Bestfit']
raw_keys = ['T','Smin','Smax','Hmin','Hmax','E1','E2','Estep']
amplitudes_raw = {k: [] for k in raw_keys}
processed_keys = ['T','Smin_avg','Smin_dev','Smax_avg','Smax_dev','Hmin_avg','Hmin_dev','Hmax_avg','Hmax_dev','E1_avg','E1_dev','E2_avg','E2_dev','Estep_avg','Estep_dev']
amplitudes_proc = {k : [] for k in processed_keys} 

fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex = 'col', figsize=(6,14)) 

# Physical constants.

F = 96485.3329
mass = 0.9*0.004203 # Mass of carbon active material, 90% loading

T_list = ['.']
lower_df = pd.DataFrame()
upper_df = pd.DataFrame()

# plotting.

for index,(directory) in enumerate(temp_dirs):
    file_list = os.listdir(directory)
    entropy_file_dict= {f.split('_')[1]: directory + f for f in file_list if f.endswith('entropy.csv')}
#    basytec_file_dict = {f.split('_')[1]: directory + f for f in file_list if f.endswith('txt')}
    for k in (entropy_file_dict):
        print(entropy_file_dict[k])
        entropy_file = entropy_file_dict[k]
#        basytec_file = basytec_file_dict[k]
        print(entropy_file)
        df_e = pd.read_csv(entropy_file,encoding='latin')
#        df_b = pd.read_csv(basytec_file,skiprows=12,encoding='latin')
#        mass = df_b['Ah[Ah]'].iloc[-1]/df_b['Ah[Ah/kg]'].iloc[-1]
#        df_e['Capacity_norm']=df_e['Charge/Discharge [mAh]']/mass
        T_K = 288 # T in Kelvin
        T = str(15)
        
        df_e['SOC'] = df_e['Charge/Discharge [mAh]']/(df_e['Charge/Discharge [mAh]'].iloc[-1]) 

        df_e['M1 Enthalpy [J mol-1]'] = - F * df_e['OCV [V]   '] + T_K * df_e['M1 Entropy [J mol-1 K-1]']
        df_e['M2 Enthalpy [J mol-1]'] = - F * df_e['OCV [V]   '] + T_K * df_e['M2 Entropy [J mol-1 K-1]']
        df_e['M3 Enthalpy [J mol-1]'] = - F * df_e['OCV [V]   '] + T_K * df_e['M3 Entropy [J mol-1 K-1]']
        df_e['M4 Enthalpy [J mol-1]'] = - F * df_e['OCV [V]   '] + T_K * df_e['M4 Entropy [J mol-1 K-1]']

        df_e['M3 Enthalpy_Lower'] = df_e['M3 Entropy_Lower [J mol-1 K-1]'] * T_K
        df_e['M3 Enthalpy_Upper'] = df_e['M3 Entropy_Upper [J mol-1 K-1]'] * T_K

        # Evaluation of amplitudes begins here.

        lower_df = df_e[df_e['SOC'].between(0.25,0.53)]
        upper_df = df_e[df_e['SOC'].between(0.53,0.7)]

        lower_v = df_e[df_e['SOC'].between(0.34,0.36)]
        upper_v = df_e[df_e['SOC'].between(0.64,0.66)]
        v_min = lower_v['OCV [V]   '].mean()
        v_max = upper_v['OCV [V]   '].mean()
        v_amp = v_max - v_min
            
        print('********T = ', str(T), ' min V = ', v_min, ' max V =', v_max, ' amplitude = ', v_amp, ' V')
        
        ax1.tick_params(which='both', length=10, width=1.5, direction='in',top=True,right=True)
        ax1.tick_params(which='minor', length=5, width=1.0, direction='in',top=True,right=True,left=True,bottom=True)

        ax1.set_ylabel('${\Delta}H$ / kJ mol$^{-1}$')
        
        for method in ['M1']:
            df_e['S'] = cumtrapz(y=df_e['%s Entropy [J mol-1 K-1]'%method].values,x=df_e['Charge/Discharge [mAh]'].values,initial=0)
            df_e['H'] = cumtrapz(y=df_e['%s Enthalpy [J mol-1]'%method].values,x=df_e['Charge/Discharge [mAh]'].values,initial=0)
            
#            ax1.plot(df_e['Charge/Discharge [mAh]'][1:-2]/mass,-df_e['%s Enthalpy [J mol-1]'%method][1:-2]/96.485,linestyle='-',markersize=5,marker='o',label='Enthalpy')
#            ax1.plot(df_e['Charge/Discharge [mAh]'][1:-2]/mass,df_e['%s Entropy [J mol-1 K-1]'%method][1:-2]*288/96.485,linestyle=':',markersize=5,marker='^',color='grey',label='Entropy')
            ax1.plot(df_e['Charge/Discharge [mAh]'][1:-2]/mass,df_e['H'][1:-2]*288/96.485,linestyle=':',markersize=5,marker='^',color='grey',label='Enthalpy')
            ax2.plot(df_e['Charge/Discharge [mAh]'][1:-2]/mass,df_e['S'][1:-2]*288/96.485,linestyle=':',markersize=5,marker='^',color='grey',label='Entropy')            
#            ax2.plot(df_e['Charge/Discharge [mAh]'][1:-2]/mass,df_e['%s Entropy [J mol-1 K-1]'%method][1:-2]*288/96.485,linestyle='-',markersize=5,marker='^',color='grey')
#            ax2.plot(df_e['Charge/Discharge [mAh]'][1:-2]/mass,df_e['S'][1:-2]*288/96.485,linestyle='-',markersize=5,marker='^',color='grey')            
#            ax1.plot(df_e['OCV [V]   '][1:-2]*1000,-df_e['%s Enthalpy [J mol-1]'%method][1:-2]/96.485,linestyle='-',markersize=5,marker='o',label='Enthalpy')
#            ax1.plot(df_e['OCV [V]   '][1:-2]*1000,df_e['%s Entropy [J mol-1 K-1]'%method][1:-2]*288/96.485,linestyle=':',markersize=5,marker='^',color='grey',label='Entropy')
#            ax2.plot(df_e['OCV [V]   '][1:-2]*1000,df_e['%s Entropy [J mol-1 K-1]'%method][1:-2]*288/96.485,linestyle=':',markersize=5,marker='^',color='grey',label='Entropy')            
#            ax2.plot(df_e['Charge/Discharge [mAh]'][1:-1]/mass,df_e['Raw Entropy [J mol-1 K-1]'][1:-1]*1000,linestyle='-',markersize=5,marker='o')            
        new_df = pd.DataFrame()
        new_df['OCV'] = df_e['OCV [V]   ']*1000
        df_e['G'] =  cumtrapz(y=-df_e['OCV [V]   '].values*1000,x=df_e['Charge/Discharge [mAh]'].values,initial=0)        
        new_df['Cap (mAh/g)'] = df_e['Charge/Discharge [mAh]']/mass
        new_df['deltaH (kJ/mol)'] = df_e['%s Enthalpy [J mol-1]'%method]
        new_df['deltaS (J/mol/K)'] = df_e['%s Entropy [J mol-1 K-1]'%method]
        new_df.to_csv('output.csv')
            
#        ax3.plot(df_e['Charge/Discharge [mAh]']/mass,df_e['OCV [V]   ']*1000,label='T = '+str(T) + r'$^{o}C$',linestyle='-',markersize = 5,marker='o',lw=2.5)
        ax3.plot(df_e['Charge/Discharge [mAh]']/mass,df_e['G'],label='T = '+str(T) + r'$^{o}C$',linestyle='-',markersize = 5,marker='o',lw=2.5)        

#        ax1.plot(df_e['SOC'][1:-1],df_e['M2 Enthalpy [J mol-1]'][1:-1],label='M2',linestyle='--',color=colors[-index-1],markersize=3)
#        ax1.plot(df_e['SOC'][1:-1],df_e['M3 Enthalpy [J mol-1]'][1:-1],label='M3',linestyle='--',color=colors[-index-1],markersize=3)
#        ax1.plot(df_e['SOC'][1:-1],df_e['M4 Enthalpy [J mol-1]'][1:-1],label='M4',linestyle='--',color=colors[-index-1],markersize=3)
        lab='sodiation'

        ax1.set_ylabel('-' + r'${\Delta}H$' +' / eV per Na')
        ax2.set_ylabel('-' + r'$T{\Delta}S$' +' / eV per Na')        
#        ax3.set_ylabel('OCV vs. Na / V')
        ax3.set_ylabel('dQ/dV (arb. units)')
#        ax1.errorbar(x=df_e['SOC'][1:-1],y=T * df_e['M3 Entropy [J mol-1 K-1]'][1:-1]/1000,yerr=[T * df_e['M3 Entropy_Lower [J mol-1 K-1]'][1:-1]/1000,T * df_e['M3 Entropy_Upper [J mol-1 K-1]'][1:-1]/1000],label='T${\Delta}S$ '+lab,marker='^',linestyle='--',elinewidth=2,color=colors[index+3],markersize=3)

#        ax2.plot(df_e['SOC'][1:-1],df_e['M2 Entropy [J mol-1 K-1]'][1:-1]*1000,label='M2',linestyle='--',color=colors[index+1],markersize=3)
#        ax2.plot(df_e['SOC'][1:-1],df_e['M3 Entropy [J mol-1 K-1]'][1:-1]*1000,label='M3',linestyle='--',color=colors[index+1],markersize=3)
#        ax2.plot(df_e['SOC'][1:-1],df_e['M4 Entropy [J mol-1 K-1]'][1:-1]*1000,label='M4',linestyle='--',color=colors[index+1],markersize=3)
        
#        ax1.set_ylim([-95,5])
#        ax2.set_ylim([-25,75])

#        ax2.legend(loc=1)

#        ax3.plot(df_e['SOC'],df_e['OCV [V]   '],label=lab,linestyle='--',color=colors[index],markersize = 3,lw=2.5)        

        ax2.tick_params(which='both', length=10, width=1.5, direction='in',top=True,right=True)
        ax2.tick_params(which='minor', length=5, width=1.0, direction='in',top=True,right=True,left=True,bottom=True)
        ax3.tick_params(which='both', length=10, width=1.5, direction='in',top=True,right=True)
        ax3.tick_params(which='minor',
    length=5, width=1.0, direction='in',top=True,right=True,left=True,bottom=True)
#       df_e.to_csv('data_entropy_%s.csv' % var)    
#        ax3.set_xlabel('Capacity (mAh/g)')
        ax3.set_xlabel('Voltage (V)')
        left, bottom, width, height = [0.32, 0.18, 0.46, 0.3] #


        dqdv = DQDV()
        dqdv.OCV_to_dQdV()
        ocv_df = dqdv.ocv_df_dis
#        ax3.plot(ocv_df['OCV'],ocv_df['dQdV'],label='dQ/dV, sodiation')
#        ax3.set_ylim([0.05,0.24])
#        for n,ax in enumerate([ax1,ax3]):
#            ax.text(-0.2,0.98,'('+string.ascii_lowercase[n]+')',transform=ax.transAxes,size=18,weight='demi')
#        ax3.legend(loc=0)
#    ax1.legend()


# ax1.set_ylim([-100,2])
# ax2.set_ylim([-5,65])
# ax3.set_ylim([-0.01,1.5])
# ax1.set_ylim([-0.01,1.5])

ax1.legend()
ax3.legend()
# plt.legend()

# ax2.legend()
# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines+ lines2, labels + labels2)

#    axes = (ax1,ax2,ax3) for cap in caps:
#cap.set_markeredgewidth(1) plt.tight_layout()
#plt.savefig('Entropy.pdf',dpi=600) plt.show()



plt.tight_layout()
plt.savefig(figures_path+'Enthalpy_entropy_ocv_sodiation.png')
plt.show()

