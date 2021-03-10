from __future__ import division
from sys import argv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import exp,log
from scipy.special import factorial
from time import sleep
import argparse
import matplotlib as mpl
import string

'''
This module is imported by the "leiva_2step" script and its only purpose is to plot things. There are various different variations: a whole series of individual png files plotting one variable after another (function plotter), or a stacked set of some of the most important plots "column_plotter". "int_plotter" is just how the individual interactions E0prime, gprime, delta1prime vary as a function of occupation x.

Please refer to the documentation for the leiva_2step script for more information about the other variables.
'''

# var_list = ['x','xmobile','dS','dSmob','S','V','mu','dxdmu', 'dxmobdmu', 'dH', 'H', 'dG', 'G', 'n1', 'n2'] # All the plots
# var_names = {'x':'x','xmobile':'x$_{r}$','dS':'dS/dx','dSmob':'dS/dx$_{r}$','S':'S','V':'V','mu':r'$\mu$','dxdmu':r'dx/d$\mu$', 'dxmobdmu':r'dx$_{r}$/d$\mu$', 'dH':'dH/dx', 'H':'H', 'dG':'dG/dx', 'G':'G', 'n1':'n1', 'n2':'n2'} # Presentation of variables on plots.
var_list = ['x','dS','S','VkT','VV','mu','dxdmu','dmudx','dH','g1','g2','E0','G','H'] # All the plots
var_names = {'x':'x','dS':'dS/dx','S':'S','VkT':'V/kT','VV':'V/V','mu':r'$\mu$','dxdmu':r'dx/d$\mu$','dmudx':r'd$\mu$/dx','dH':'d$H$/d$x$','g1':'g1','g2':'g2','E0':'e0','G':'G','H':'H'} # Presentation of variables on plots.
# units = {'x':'','xmobile':'','dS' : 'J mol$^{-1}$ K$^{-1}$', 'dSmob' : 'J mol$^{-1}$ K$^{-1}$', 'S' : 'J mol$^{-1}$ K$^{-1}$', 'V' : 'V vs. Li/Li$^{+}$', 'mu' : 'eV', 'dxdmu' : 'eV$^{-1}$', 'dxmobdmu' : 'eV$^{-1}$', 'dH' : 'kJ mol$^{-1}$', 'H' : 'kJ mol$^{-1}$', 'dG' : 'kJ mol$^{-1}$', 'G' : 'kJ mol$^{-1}$', 'n1' : '', 'n2' : ''} # Units for all the plots.
units = {'x':'','dS' : '$2Mk$', 'S' : 'J mol$^{-1}$ K$^{-1}$', 'VkT' : 'kT','VV' : 'V vs. Li', 'mu' : 'eV', 'dxdmu' : 'eV$^{-1}$','dmudx':'a.u.', 'dH' : 'a.u.','g1':'kT' ,'g2':'kT','E0':'kT','G':'kJ mol$^{-1}$','H':'kJ mol$^{-1}$'} # Units for all the plots.

# plt.style.use('classic')
mpl.rcParams['lines.linewidth'] = 2.5

font = {'size': 24}

mpl.rc('font', **font)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

class Plotting():
    def __init__(self,df_dict,long_dict):
        self.df_dict = df_dict
        self.long_dict = long_dict
        self.T = 298 # Temperature in kelvin
        self.output_path = '../output_2l/'
        
    def plotter(self,loc_value):
        for key,value_list in self.long_dict.items():
            if key == 'n1':
                for value in value_list:
                    suffix = value[0].split('_')[-1]
                    plt.plot(self.df_dict['x'],self.df_dict[key+'_'+str(value[1])], label=str(value[1]) + ', n1',linewidth=1.5)
                    plt.plot(self.df_dict[value[1]]['x'],self.df_dict[key+'_'+str(value[1])], label=str(value[1]) + ', n2',linewidth=1.5)
            elif key != 'n2':
                for value in value_list:
#                plt.plot(df_dict[value[1]]['x'],df_dict[value[1]][str(value[0])], label=str(value[1]))
                    plt.plot(self.df_dict['x'],self.df_dict[key+'_'+str(value[1])], label=r'$\alpha$ = '+str(value[1]),linewidth=1.5)
            plt.xlabel('$x$')
            plt.ylabel(str(var_names[key]) + ' / ' + str(units[key]))
#            if key != 'S':
#                plt.legend(loc=loc_value,fontsize=11)
#        plt.ylim([0,8])
            plt.tight_layout()
#            plt.xlim([0.0,0.15])
            plt.savefig(self.output_path + '%svsx.png'% str(key),dpi=300)
            plt.clf()
            if key == 'n1':
                for value in value_list:
                    suffix = value[0].split('_')[-1]
                    plt.plot(self.df_dict[value[1]]['n1_' + suffix], self.df_dict[value[1]]['H_' + suffix], label=str(value[1]) + ', n1',linewidth=0.75)
                    plt.plot(self.df_dict[value[1]]['n2_' + suffix], self.df_dict[value[1]]['H_' + suffix], label=str(value[1]) + ', n2',linewidth=0.75)
            plt.xlabel('H / kJ mol-1')
            plt.ylabel(str(var_names[key]) + ' / ' + str(units[key]))
            plt.clf()
            if key != 'n1':
                for value in value_list:
                    suffix = value[0].split('_')[-1]
                    plt.plot(self.df_dict['VV_'+str(value[1])],self.df_dict[key+'_'+str(value[1])], label=str(value[1]),linewidth=1.5)
                    plt.legend()
            plt.xlabel('Voltage / V')
            plt.ylabel(str(var_names[key]) + ' / ' + str(units[key]))                       
            plt.tight_layout()
#            plt.xlim([0.0,0.15])
            plt.savefig(self.output_path + '%svsV.png'% str(key),dpi=300)
            plt.clf()                    

#        voltages=self.long_dict['VV']
#        voltage_list=[]
#        for entry in voltages:
#            voltage_list.append(entry[0])
#        print voltage_list

#        for key,value_list in self.long_dict.iteritems():    
#            for n,value in enumerate(value_list):
#                plt.plot(self.df_dict[value[1]][voltage_list[n]][1:-2],self.df_dict[value[1]][str(value[0])][1:-2], label=str(value[1]),linewidth=1.5)
#            plt.xlabel('E / V vs. Li')
#            plt.xlim(0.05,0.25)
#            plt.ylabel(str(var_names[key]) + ' / ' + str(units[key]))
#            plt.legend(loc=loc_value,fontsize=16)
#            plt.ylim([0,8])
#            plt.savefig('output/%svsV.png'% str(key),dpi=300)
#            plt.clf()
    def alt_plotter(self,loc):
        for key, df in self.df_dict.items():
            plt.plot(df['V_'+str(key)],df['dxdmu_'+str(key)],label=key)
        plt.legend(loc=0)
        plt.show()

    def twobytwo(self):
        f, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2, sharex='col', figsize =(7,7))

        ax5 = ax1.twinx()
        ax6 = ax2.twinx()
        axes=(ax1,ax3,ax2,ax4,ax5,ax6)
        xkeys = zip(*self.long_dict['x'])[1] # Get the values to iterater over
        for i,k in enumerate(reversed(xkeys)):
            df=self.df_dict
            df_r = df[df['x'] < 0.1]
   
            ax5.plot(df_r['x'],-self.T * df_r['S_' + str(k)]/1000,label=r'$\alpha$'+ '= %.3g' % float(k),color=colors[i])
            ax5.legend()
            ax1.plot(df_r['x'],df_r['H_' + str(k)],label=r'$\alpha$'+ '= %.3g' % float(k),color=colors[i+1])
            ax3.plot(df_r['x'],df_r['G_' + str(k)],label=r'$\alpha$'+ '= %.3g' % float(k),color=colors[i+2])
            ax2.plot(df_r['x'],-self.T * df_r['dS_' + str(k)]/1000,label=r'$\alpha$'+ '= %.3g' % float(k),color=colors[i])
            ax6.plot(df_r['x'],df_r['dH_' + str(k)],label=r'$\alpha$'+ '= %.3g' % float(k),color=colors[i+1])
            ax4.plot(df_r['x'],df_r['mu_' + str(k)],label=r'$\alpha$'+ '= %.3g' % float(k),color=colors[i+2])
#            ax2.plot(df['dxdmu_' + str(k)],df['VV_'+str(k)],label='alpha' + '= %.3g' % float(k),color=colors[3*i])
 #           ax3.plot(df['VV_' + str(k)],df['dxdmu_'+str(k)],label='alpha' + '= %.3g' % float(k),color=colors[3*i])
 #           ax4.plot(df['VV_' + str(k)],df['dxdmu_'+str(k)],label='alpha' + '= %.3g' % float(k),color=colors[3*i])
#            ax3.plot(df['VV_' + str(k)],df['dxdmu_'+str(k)],label='alpha' + '= %.3g' % float(k),color=colors[3*i])
#            ax2.plot(df['x'],df['dH_'+str(k)],label='alpha'+'= %.3g' % float(k))
#            ax4.plot(df['x'],df['dS_'+str(k)],label=r'$\alpha$'+ '= %.3g' % float(k))
#            ax2.plot(df['x'],df['dH_'+str(k)],label='alpha'+'= %.3g' % float(k),color=colors[3*i])
#            ax4.plot(df['x'],df['dS_'+str(k)],label=r'$\alpha$'+ '= %.3g' % float(k),color=colors[3*i])
        ax1.set_ylabel('H')
        ax5.set_ylabel('-TS')
        ax3.set_ylabel('G')
        
#        ax2.set_ylabel('Voltage / V vs. Li')
        ax3.set_xlabel('Li occupation, $x$')
        ax4.set_xlabel('Li occupation, $x$')                
#        ax2.set_xlabel('dQ/dV')
#        ax3.set_ylabel('d$x$/d$V$ (arb. units)')        
#    ax1.get_yaxis().set_label_coords(-0.18,0.5)
#    ax2.get_yaxis().set_label_coords(1.22,0.5)
#    ax3.get_yaxis().set_label_coords(-0.18,0.5)
#    ax4.get_yaxis().set_label_coords(-5,5)
        ax2.yaxis.set_label_position('right')
        ax4.yaxis.set_label_position('right')        
        ax2.yaxis.tick_right()
        ax4.yaxis.tick_right()
#        ax1.xaxis.tick_top()
#        ax3.xaxis.tick_top()
        ax2.yaxis.set_ticks_position('both')
 #       ax4.yaxis.set_ticks_position('both')
 #       ax1.xaxis.set_ticks_position('both')
 #       ax3.xaxis.set_ticks_position('both')
#        ax2.set_ylabel('d$H$/d$x$ / kJ mol$^{-1}$')
#        ax2.set_ylabel('d$H$/d$x$ / kT')
#        ax4.set_ylabel('d$S$/d$x$ / J mol$^{-1}$ K$^{-1}$')

#        ax4.set_ylim([-5,5])
 #   ax2.set_ylim([-0.5,6.5])
 #   ax1.set_yticks(np.arange(3.7,4.3,0.1))
        ax1.set_xlim([0,0.1])
        ax5.set_xlim([0,0.1])
        ax1.autoscale(enable=True,axis='y')
#        ax7.set_xlim([0,0.2])        
#        ax1.set_ylim([-0.02,0.25])
 #       ax2.set_ylim([-0.02,0.25])        
#        ax3.set_xlim([0.15,0.22])
#        ax2.set_ylim([-20,-5])
#        ax4.set_ylim([-15,75])
#        ax1.set_xlim([0.05,0.25])
#        ax3.set_xlim([0.05,0.25])
#        ax1.set_ylim([-0.04,0.23])
#        ax3.set_ylim([-0.5,13])
#        ax2.set_ylim([-17,-8])

        
        ax3.set_xlabel('Voltage vs. Li / V')
#        ax1.xaxis.set_label_position('top')
#        ax2.xaxis.set_label_position('top')
        ax4.set_xlabel('Li content, $x$')
#        ax4.set_xlabel('Voltage vs. Li / V')
#    handles,labels = ax1.get_legend_handles_labels()
#    labels,handles = zip(*sorted(zip(labels,handles), key=lambda t:t[0])
    
        ax4.legend(loc=0,fontsize=12,handletextpad=0.1)
#        for n,ax in enumerate(axes):
#            if ax == ax1 or ax == ax3:
#                ax.text(0.83,0.87,'('+string.ascii_lowercase[n]+')',transform=ax.transAxes,size=15,weight='demi')
#            else:
#                ax.text(0.05,0.87,'('+string.ascii_lowercase[n]+')',transform=ax.transAxes,size=15,weight='demi')
#    ax1.set_ylim([3.65,4.25])
#    ax3.set_ylim([-0.5,10.5])
        plt.tight_layout()
        f.subplots_adjust(hspace=0)
        plt.savefig(self.output_path + 'fig2by2.pdf',dpi=400)
        plt.show()

    def onebytwo(self):
        f, (ax1, ax2) = plt.subplots(1,2, sharex='col', figsize =(7,4))
        axes=(ax1,ax2)
        xkeys = zip(*self.long_dict['x'])[1] # Get the values to iterater over
        for i,k in enumerate(reversed(xkeys)):
            df=self.df_dict
            ax1.plot(df['VV_' + str(k)],df['E0_' + str(k)],label=r'$\alpha$'+ '= %.3g' % float(k),color=colors[3*i])
            ax2.plot(df['x'],df['E0_' + str(k)],label=r'$\alpha$' + '= %.3g' % float(k),color=colors[3*i])
        ax1.set_ylabel("$E\'_{0}$ / kT")
        ax2.set_ylabel("$E\'_{0}$ / kT")        
#    ax1.get_yaxis().set_label_coords(-0.18,0.5)
#    ax2.get_yaxis().set_label_coords(1.22,0.5)
#    ax3.get_yaxis().set_label_coords(-0.18,0.5)
#    ax4.get_yaxis().set_label_coords(-5,5)
        ax2.yaxis.set_label_position('right')        
        ax2.yaxis.tick_right()

#        ax4.set_ylim([-5,5])
 #   ax2.set_ylim([-0.5,6.5])
 #   ax1.set_yticks(np.arange(3.7,4.3,0.1))
        ax1.set_xlim([0.05,0.25])
        ax1.set_ylim([-7.5,-2.5])
        ax2.set_ylim([-7.5,-2.5])

        ax1.set_xlabel('Voltage vs. Li / V')
#        ax1.xaxis.set_label_position('top')
#        ax2.xaxis.set_label_position('top')
        ax2.set_xlabel('Li content, $x$')
        
#    handles,labels = ax1.get_legend_handles_labels()
#    labels,handles = zip(*sorted(zip(labels,handles), key=lambda t:t[0])
    
        ax1.legend(loc=3,fontsize=12,handletextpad=0.1)
        for n,ax in enumerate(axes):
            if ax == ax1:
                ax.text(0.05,0.90,'('+string.ascii_lowercase[n]+')',transform=ax.transAxes,size=15,weight='demi')
            else:
                ax.text(0.07,0.90,'('+string.ascii_lowercase[n]+')',transform=ax.transAxes,size=15,weight='demi')
#    ax1.set_ylim([3.65,4.25])
#    ax3.set_ylim([-0.5,10.5])
        plt.tight_layout()
        f.subplots_adjust(hspace=0)
        plt.savefig(self.output_path + 'fig1by2.pdf',dpi=400)
        plt.show()
        
    def getkey(self,dict_iteritems):
        return(float(dict_items[0].split('_')[1]))

    def int_plotter(self):
        keys = list(self.df_dict)
        for interaction in ['delta1','g','E0']:
            for value in keys:
                df = self.df_dict[value]
                plt.plot(df['x'],df[interaction + '_' + value],label=interaction + ' = ' + value)
            plt.legend(fontsize=16)
            plt.savefig('interaction_' + interaction + '.png')             
                         
    def column_plot(self):
        f, ((ax1, ax2,ax3)) = plt.subplots(3,1, figsize=(4.5,9),sharex='col')
        axes = (ax1, ax2, ax3)
        key_vals = [k.split('_')[1] for k in self.df_dict.keys() if k.startswith('g1')]
        for k in key_vals:    
            lab = '%.3f' % (float(k))
#        lab = '$J_{2}$ = ' +k
            ax1.plot(self.df_dict['x'],self.df_dict['VV_' + str(lab)],label=lab,linewidth=0.75)
            ax2.plot(self.df_dict['x'],self.df_dict['dxdmu_'+str(lab)],label=lab,linewidth=0.75)
            ax3.plot(self.df_dict['x'],self.df_dict['dS_'+str(lab)],label=lab,linewidth=0.75)
        ax1.set_ylabel('Voltage / V')
        ax2.set_ylabel('d$x$/d$V$')
        ax3.set_ylabel('d$S$/d$x$ / J mol-1 K-1')
        ax1.get_yaxis().set_label_coords(-0.1,0.5)
        ax2.get_yaxis().set_label_coords(-0.1,0.5)
        ax3.get_yaxis().set_label_coords(-0.1,0.5)
        ax1.yaxis.set_label_position('right')
        ax1.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.tick_right()
        ax3.yaxis.set_label_position('right')
        ax3.yaxis.tick_right()
#   ax1.set_ylim([-0.05,1.05])
#   ax2.set_ylim([-0.05,11.3])
#   ax3.set_ylim([-25,38])
        for n,ax in enumerate(axes):
            if ax == ax1:
                ax.text(0.03,0.1,'('+string.ascii_lowercase[n]+')',transform=ax.transAxes,size=15,weight='demi')
            else:
                ax.text(0.03,0.87,'('+string.ascii_lowercase[n]+')',transform=ax.transAxes,size=15,weight='demi')
#        ax1.set_xlim(0.05,0.25)
#        ax2.set_xlim(0.05,0.25)
#        ax3.set_xlim(0.05,0.25)        
        ax3.set_xlabel('x')

        ax1.legend(loc=0,fontsize=13)
        plt.tight_layout()
        f.subplots_adjust(hspace=0)
        plt.savefig(self.output_path + 'column_fig.png',dpi=400)
        plt.show()

    def double_plot(self):
        f, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(9,9),sharex='col')
        axes = (ax1, ax2, ax3, ax4)
        for k, df in sorted(self.df_dict.items(),key=self.getkey):
            lab = '%.3f' % (float(k))
#        lab = '$J_{2}$ = ' +k
            ax1.plot(df['VV_' + str(k)],df['x'],label=lab,linewidth=0.75)
            ax2.plot(df['VV_' + str(k)],df['dxdmu_'+str(k)],label=lab,linewidth=0.75)
            ax3.plot(df['x'],df['dS_'+str(k)],label=lab,linewidth=0.75)
            ax4.plot(df['x'],df['dS_'+str(k)],label=lab,linewidth=0.75)
        ax1.set_ylabel('Li content, $x$')
        ax2.set_ylabel('d$x$/d$V$ / kT')
        ax3.set_ylabel('d$S$/d$x$ / 2Mk')
        ax1.get_yaxis().set_label_coords(-0.1,0.5)
        ax2.get_yaxis().set_label_coords(-0.1,0.5)
        ax3.get_yaxis().set_label_coords(-0.1,0.5)
        ax1.yaxis.set_label_position('right')
        ax1.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.tick_right()
        ax3.yaxis.set_label_position('right')
        ax3.yaxis.tick_right()
#   ax1.set_ylim([-0.05,1.05])
#   ax2.set_ylim([-0.05,11.3])
#   ax3.set_ylim([-25,38])
        for n,ax in enumerate(axes):
            if ax == ax1:
                ax.text(0.03,0.1,'('+string.ascii_lowercase[n]+')',transform=ax.transAxes,size=15,weight='demi')
            else:
                ax.text(0.03,0.87,'('+string.ascii_lowercase[n]+')',transform=ax.transAxes,size=15,weight='demi')
        ax1.set_xlim(0.05,0.25)
        ax2.set_xlim(0.05,0.25)
        ax3.set_xlim(0.05,0.25)        
        ax3.set_xlabel('Voltage / V vs. Li')

        ax1.legend(loc=0,fontsize=11)
        plt.tight_layout()
        f.subplots_adjust(hspace=0)
        plt.savefig(self.output_path + 'column_fig.png',dpi=400)
        plt.show()
