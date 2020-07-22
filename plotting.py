'''
This is a collection of scripts that runs the leiva3.py program and performs various plotting functions. The plotting function that is used for each figure is also displayed in the comments beside each function.

Note that to run this, you need to use Python 2.7 along with various modules: matplotlib, numpy, pandas and scipy. All of these modules are included with the Anaconda package.

In this and the leiva3.py script (the one that does the maths), you enter the input variables as a sequence of command line arguments. Arguments that take a single value are kept fixed. Please see the __main__ function for more information, or use help() within the interactive version of Python. Multiple arguments are assigned to the variable that is changed in the plot. You alos currently have to highlight this value using the argument --label.

Each of the important plotting functions take three arguments, as follows:

df_dict: this is a dictionary of Pandas dataframes generated by the leiva3.py script.
long_dict: this dictionary contains information on the input variables, which are generated on the command line.
label: the tells the plot which variable to label as changing in the legend.

For your convenience I put in the combination of command line arguments that will reproduce each plot. I just run the commands directly from the terminal. Note I highlight here the ones where the variables could be changed interactively:

Figure 7: use sub_plotter_voltage

python plotter.py --E0 4.1 --J1 0 12.5 25 30 45 --label J1 --nargs 5

Figure 8: use sub_plotter

python plotter.py --E0 4.1 --J1 0 12.5 25 30 45 75 900 --label J1 --nargs 7

Figure 9: use column_plot

python plotter.py --E0 4.1 --J1 30 --J2 -2.5 -1.25 0 1.25 2.50 --label J2 --nargs 5

Figure 10: use column_plot

python plotter.py --E0 4.1 --J1 30 --J2 -1.25 --label delta --nargs 5 --delta 0 0.25 0.75 1.25 1.75

Figure 11: use double_column_plot

python plotter.py --E0 4.1 --J1 30 --J2 -1.25 --label delta --nargs 5 --delta 1.25
'''
from __future__ import division
from sys import argv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import exp,log
from scipy.misc import factorial
from time import sleep
import argparse
import matplotlib as mpl
import string
from leiva3 import *

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
label_dict = {'overlit' : r'$y$','J1':r'$J_{1}$','J2':r'$J_{2}$','E0':r'$E_{0}$','delta':r'$\delta$'}

Nfirst = 4 # Number of nearest neighbours in the lattice. 
Nsecond = 12 # Number of second nearest neighbours (double counting accounted later)

def column_plot(df_dict,long_dict,label):
    # Figures 9 and 10 use this format.
    f, ((ax1, ax2,ax3)) = plt.subplots(3,1, figsize=(4.5,9),sharex='col')
    axes = (ax1, ax2, ax3)
    print list(long_dict.keys())
    for k, df in sorted(df_dict.iteritems(),key=getkey):
        lab = label + ' = %.1f' % (float(k))
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

def sub_plotter_voltage(df_dict,long_dict,label):
    # Format adopted by Figure 7.
    f, ((ax1, ax2)) = plt.subplots(1,2, sharex='col', figsize =(7,4))
    axes=(ax1,ax2)
    print list(long_dict.keys())
    for k, df in sorted(df_dict.iteritems(),key=getkey):
        lab = label + ' = %.1f' % (float(k))
#        lab = '$J_{2}$ = ' +k
        ax1.plot(df['V_' + str(k)],df['dxdmu_'+str(k)],label='%.3g' % float(k))
        ax2.plot(df['V_' + str(k)],df['dS_'+str(k)],label='%.3g' % float(k))
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

def sub_plotter_rem(df_dict,long_dict,label):
    # Alternative version of the subplotter script. Instead plots with respect to removable Li population, rather than the total Li population.
    f, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2, sharex='col', figsize =(7,6))
    axes=(ax1,ax3,ax2,ax4)
    print list(long_dict.keys())
    for k, df in sorted(df_dict.iteritems(),key=getkey):
        lab = label + ' = %.1f' % (float(k))
        ax1.plot(df['xmobile'],df['V_' + str(k)],label=lab)
        ax3.plot(df['xmobile'],df['S_' +str(k)],label=lab)
        ax2.plot(df['V_' + str(k)],df['dxmobdmu_'+str(k)],label=lab)
        ax4.plot(df['V_' + str(k)],df['dSmob_'+str(k)],label=lab)
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

def double_column_plot(df_dict,long_dict,label):
    # Format adopted by Figure 10.
    f, ((ax1, ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2, figsize=(7,8), sharex='col')
    axes=(ax1,ax2,ax3,ax4,ax5,ax6)
    print list(long_dict.keys())
    for k, df in sorted(df_dict.iteritems(),key=getkey):
        overlit_frac = float(k)
        for k, df in sorted(df_dict.iteritems(),key=getkey):
            lab = label + ' = %.1f' % (float(k))
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


def plotter(df_dict, long_dict,loc_value):
    # Produces single y as a function of x plots of all relevant thermodynamic varaibles (not used for the paper.
    for key,value_list in long_dict.iteritems():
        print 'key,value_list', key, value_list
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
    print voltage_list

    for key,value_list in long_dict.iteritems():    
        for n,value in enumerate(value_list):
            plt.plot(df_dict[value[1]][voltage_list[n]][1:-2],df_dict[value[1]][str(value[0])][1:-2], label=str(value[1]))
        plt.xlabel('E / V vs. Li')
        plt.ylabel(str(var_names[key]) + ' / ' + str(units[key]))
        plt.legend(loc=loc_value,fontsize=16)
        plt.ylim([0,8])
        plt.savefig('output/%svsV.png'% str(key),dpi=300)
        plt.clf()

def alt_plotter(df_dict,long_dict,loc):
    # Alternative format of individual plots. Not used in the paper.
    for key, df in df_dict.iteritems():
        plt.plot(df['V_'+str(key)],df['dxdmu_'+str(key)],label=key)
    plt.legend(loc=0)
    plt.show()
    print list(df_dict)

def sub_plotter(df_dict,long_dict,label):
    # Used to produce Figure 8.
    f, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2, sharex='col', figsize =(7,7))
    axes=(ax1,ax3,ax2,ax4)
    print list(long_dict.keys())
    for k, df in sorted(df_dict.iteritems(),key=getkey):
        lab = label + ' = %.1f' % (float(k))
        ax1.plot(df['x'],df['V_' + str(k)],label=lab)
        ax3.plot(df['x'],df['dxdmu_'+str(k)],label=lab)
        ax2.plot(df['x'],df['S_'+str(k)],label=lab)
        ax4.plot(df['x'],df['dS_'+str(k)],label=lab)
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
    ax1.set_ylim([3.65,4.25])
    ax3.set_ylim([-0.5,10.5])
    plt.tight_layout()
    f.subplots_adjust(hspace=0)
    plt.savefig('output/modelfig1.png',dpi=400)
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
    label = args.label

    arg_dict = dict(vars(args))
    del arg_dict['loc']
    del arg_dict['nargs']

    for key,value in arg_dict.iteritems():
        if key == 'label':
            label_vals = arg_dict[label]
        if key != 'Mprime' and key != 'label':
            while len(value) < args.nargs:
                arg_dict[key].append(value[-1])
    
    print 'Full list: ', arg_dict
            
    print 'E0 = ', E0
    print 'deltaE = ', deltaE
    
    counter = 0
    df_dict = g_zero_solution(arg_dict,counter)
    label_name = label_dict[label] # Asigns the variable to the legend.
# Active plotting scripts to be placed here.
#    column_plot(df_dict,arg_dict,label_name)
    column_plot(df_dict,arg_dict,label_name) # As requested, plots in a format similar to Figure 9.

    