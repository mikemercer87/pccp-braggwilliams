import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.interpolate import interp1d
import numpy as np

#df0=pd.read_csv('abs_mu_-1,0_1,0_0,005.out',names=['T','mu','E-mu*x','x','phi','E2','x2','Elte-mu*x_lte','x_lte','phi_lte','E_mf-mu*x_mf','x_mf','phi_mf','E_hte-me*x_hte','x_hte','phi_hte','lro','corr'],usecols=[1,3],sep='\t')

aa_left_point = 80.807 # x = 0
aa_right_point = 0 # x = 1
aabb_left_point = 42.025 # x =0
aabb_right_point = -16.12 # x = 0.5

def read_file(index):
    df = pd.read_csv('abs_-0,3_0,2_er%s_mustep0,5.out' % index,names=['mu','x','varE','varx'],usecols=[1,3,5,6],sep='\t')
    return(df)

def aa_correction(df):
    right_lim = df.iloc[-1]
    left_lim = df.iloc[0]
    right_old = right_lim['G']
    left_old = left_lim['G']
    left_new = 80.807/1000 # x = 0
    right_new = 0 # x = 1
    print(right_old,right_lim)
    df['G'] = df['G'] + df['x_real'] * (right_new - right_old) + (1-df['x_real'])* (left_new - left_old)
    plt.plot(df['x_real'],df['G'],linestyle='--',marker='o',label='aaaa',markersize=3)    
    return(df)

def aabb_correction(df):
    right_old = df.iloc[-1]['G']
    left_old = df.iloc[0]['G']
    left_new = 42.025 / 1000 # x = 0
    right_new = -16.12 / 1000 # x = 1
    df['G'] = df['G'] + 2 * df['x_real'] * (right_new - right_old) + (1-2 * df['x_real'])* (left_new - left_old)
    return(df)


#df1=pd.read_csv('mu-1,0_er13_gs1.out',names=['T','mu','E-mu*x','x','phi','E2','x2','Elte-mu*x_lte','x_lte','phi_lte','E_mf-mu*x_mf','x_mf','phi_mf','E_hte-me*x_hte','x_hte','phi_hte','lro','corr'],sep='\t')
#df2=pd.read_csv('mu-1,0_er13_gs2.out',names=['T','mu','E-mu*x','x','phi','E2','x2','Elte-mu*x_lte','x_lte','phi_lte','E_mf-mu*x_mf','x_mf','phi_mf','E_hte-me*x_hte','x_hte','phi_hte','lro','corr'],sep='\t')
# print(df0['x'].iloc[0:5])

all_labels = ['36']
all_dfs = [read_file(index) for index in all_labels]
odd_labels = []
even_labels = []
label_dict = {}
df_dict = {label : df for label,df in zip(all_labels,all_dfs)}

for n,label in enumerate(all_labels):
    label_dict[label] = label
    

for label,df in df_dict.items():
    df['x_real'] = 0.5*df['x'] + 0.5
    df['mu_realx'] = df['mu']
    df['G'] = integrate.cumtrapz(y = 2*df['mu'].values,x = df['x_real'].values,initial=0)
    df = aa_correction(df)
#    size = str(int(label_dict[label]))


df_aabb = pd.read_csv('aabb_sizedep_er208.out',names=['mu','x','varE','varx'],usecols=[1,3,5,6],sep='\t')

df_aabb['x_real'] = 0.25*df_aabb['x'] + 0.25
df_aabb['mu_realx'] = 2 * df_aabb['mu']
df_aabb['G'] = integrate.cumtrapz(y = 2*df_aabb['mu'].values,x = df_aabb['x_real'].values,initial=0)
df_aabb = aabb_correction(df_aabb)
df_ground=pd.read_csv('aa.txt')
print(list(df_ground))
plt.plot(df_ground['c'],df_ground['E(meV/6C)']/1000)

#    size = str(int(label_dict[label]))
# plt.plot(df['x_real'],df['G'],linestyle='--',marker='o',label='aa',markersize=3)
# plt.plot(df_aabb['x_real'],df_aabb['G'],linestyle='--',marker='o',label='aabb',markersize=3)

faa = interp1d(df['x_real'],df['G'],fill_value='extrapolate')
faa_gnd = interp1d(df_ground['c'],df_ground['E(meV/6C)']/1000,fill_value='extrapolate')

x_aa_new=np.linspace(0,1,10001)
x_aa_gnd_new = np.linspace(0,1,10001)

yaa_new = faa(x_aa_new)
yaa_gnd_new = faa_gnd(x_aa_gnd_new)
# mu_aa = np.gradient(yaa_new,x_aa_new[1] - x_aa_new[0])
# mu_aabb = np.gradient(yaabb_new,x_aabb_new[1] - x_aabb_new[0])
plt.plot(x_aa_new,yaa_new)
plt.plot(x_aa_gnd_new,yaa_new-yaa_gnd_new)


plt.xlabel('x in Li$_{x}$C$_{6}$')
plt.ylabel('G(x) / eV per site')
plt.legend()
plt.show()

plt.clf()
'''
x_aa_new=np.linspace(0,1,10001)
x_aabb_new = np.linspace(0,0.5,10001)

faa = interp1d(df['x_real'],df['G'],fill_value='extrapolate')
faabb = interp1d(df_aabb['x_real'],df_aabb['G'],fill_value='extrapolate')

yaa_new = faa(x_aa_new)
yaabb_new = faabb(x_aabb_new)
mu_aa = np.gradient(yaa_new,x_aa_new[1] - x_aa_new[0])
mu_aabb = np.gradient(yaabb_new,x_aabb_new[1] - x_aabb_new[0])

plt.plot(x_aa_new,-mu_aa)
# plt.plot(x_aabb_new,-mu_aabb)
plt.xlabel('x in LixC6')
plt.ylabel('-mu / eV per site')
plt.show()
'''
