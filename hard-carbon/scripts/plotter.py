import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

mpl.rcParams['lines.linewidth'] = 2.5

font = {'size': 18}

mpl.rc('font', **font)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams.update({'errorbar.capsize': 2})
mpl.rcParams['legend.handlelength'] = 0


df = pd.read_csv('hard_carbon_3cycles_00.txt',skiprows=12)
df['Cyc-Count-diff'] = df['Cyc-Count'].diff()
changes = df[df['Cyc-Count-diff']!=0].index.tolist()
changes.append(df['Cyc-Count'].index[-2])
print('changes=',changes)
last_index = 0

for cycle_no,index in enumerate(changes[2:]):
    last_index = changes[cycle_no-4]
    print(last_index,index)
    cycle_df = df.loc[last_index+1:index]
#    cycle_df = cycle_df[cycle_df['I[A/kg]'] != 0] # Exclude OCV
    cycle_df['Capacity'] = cycle_df['Ah[Ah/kg]'] - cycle_df['Ah[Ah/kg]'].iloc[last_index-index]
    cycle_df_c = cycle_df[(cycle_df['I[A/kg]']) > 9]
   
#    plt.plot(-cycle_df_c['Capacity'],cycle_df_c['U[V]'],label='Charge,'+str(cycle_no+1))

    
    cycle_df_d = cycle_df[cycle_df['I[A/kg]'] < -10]
    cycle_df_c =  cycle_df[cycle_df['I[A/kg]'] > 9]
#    plt.plot(cycle_df_d['~Time[h]'],cycle_df_d['I[A/kg]'],label='Charge,'+str(cycle_no+1),linestyle='',marker='o')
    linestyles = ['-','dashed','dashed']
    colors=['royalblue','grey','salmon']
    plt.plot(-cycle_df_d['Capacity'],cycle_df_d['U[V]'],label='Cycle '+str(cycle_no+6),color=colors[cycle_no],linestyle=linestyles[cycle_no])
    plt.plot(-cycle_df_c['Capacity'],cycle_df_c['U[V]'],color=colors[cycle_no],linestyle=linestyles[cycle_no])        
    last_index = index

plt.xlabel('Discharge capacity (mAh/g$_{HC}$)')
plt.ylabel('Voltage (V) vs. Na') 
plt.legend(handlelength=2)
plt.savefig('3_cycles.png')
plt.show()
'''
print(changes)
plot, = plt.plot(-df['Ah[Ah/kg]'].iloc[52491:],df['U[V]'].iloc[52491:])
plt.legend()
plt.show()
'''
