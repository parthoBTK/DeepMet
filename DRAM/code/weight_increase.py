# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 17:29:04 2019

@author: yliu258
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 01:36:28 2019

@author: yliu258
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 00:24:04 2019

@author: yliu258
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from cycler import cycler


plt.rc('figure', dpi=120)
plt.rc('savefig', dpi=300, bbox = 'tight', format='png')
plt.rc('pdf', fonttype = 42)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Times New Roman')

plt.rc('figure.subplot', wspace=0.1, hspace=0.1, left=0.1, top=0.9)
plt.rc('axes', titlesize = 16, labelsize = 18, linewidth = 1.5)
plt.rc('axes', prop_cycle = cycler('color', ['red','blue','green',
                                             'black', '#1F4096', '#E84A27']))
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('lines', linewidth=2, markersize=8)
plt.rc('legend', fontsize=13, framealpha=0.95, handlelength=2.2, 
       shadow=False, fancybox=False, edgecolor='k')



#for ind in range(1,num_dir+1):


#%% Plot mean error 
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
num_dir = np.array([os.path.isdir(x) for x in os.listdir()]).sum()
error = []
for ind in np.arange(1,num_dir+1):
    data_pred = np.loadtxt('omega='+str(ind)+'/model_prediction.txt', delimiter=',')
    data_test = np.loadtxt('omega='+str(ind)+'/test_data.txt', delimiter=',')
    relative_error = 100*np.abs(data_pred-data_test)/data_test
    error.append(np.max(relative_error,axis=0))
    
error = np.asarray(error)

fig2, ax2 = plt.subplots()
ax2.scatter(np.arange(1,num_dir+1), error[:,0],marker='*',label='$\mathrm{D}_1$')
ax2.scatter(np.arange(1,num_dir+1), error[:,1],marker='o', label='$\mathrm{H}_1$')
ax2.scatter(np.arange(1,num_dir+1), error[:,2],marker='D',label='$\mathrm{H}_2$')

ax2.scatter(np.arange(1,num_dir+1), error[:,3],marker='s',label='$\mathrm{H}_3$')
ax2.scatter(np.arange(1,num_dir+1), error[:,4],marker='^',label='$\mathrm{SWA}_1$')
ax2.scatter(np.arange(1,num_dir+1), error[:,5],marker='D',label='$\mathrm{SWA}_2$')

ax2.set_ylabel('Max Error (\%)')
ax2.legend(ncol=2)

ax2.xaxis.set_minor_locator(MultipleLocator())

ax2.tick_params(which='both', width=2)
ax2.set_xlabel(r'$\omega_{H_2}$')
fig2.savefig('others')

#%% Two axis
fig2, ax2 = plt.subplots()
ax2.scatter(np.arange(1,num_dir+1), error[:,0],marker='*', color='#1F4096', label='D1')
ax2.scatter(np.arange(1,num_dir+1), error[:,1],marker='o', color='r', label='H1')
#ax2.scatter(np.arange(1,6), error[:,2],marker='D',label='H2')

ax2.scatter(np.arange(1,num_dir+1), error[:,3],marker='s', color='k', label='H3')
ax2.scatter(np.arange(1,num_dir+1), error[:,4],marker='^', color='b', label='Theta1')
ax2.scatter(np.arange(1,num_dir+1), error[:,5],marker='D', color='g', label='Theta2')

ax = ax2.twinx()
ax.scatter(np.arange(1,num_dir+1), error[:,2],marker='D', color='m', label='H2')
ax.set_ylabel('Max Error (\%) (H2)', color='m')
ax.tick_params(axis='y', labelcolor='m')
#ax.set_ylim([6,12])
#ax2.set_ylim([0.25,1.75])

ax.legend()

ax2.set_ylabel('Max Error (\%)')
ax2.legend(ncol=2)

ax2.xaxis.set_minor_locator(MultipleLocator())

ax2.tick_params(which='both', width=2)
ax.tick_params(which='both', width=2)
ax2.set_xlabel('$W_{H_3}$')

#%% Two plots
fig2, ax2 = plt.subplots()
ax2.scatter(np.array([1,5,9]), error[:,0],marker='*', color='#1F4096', label='$\mathrm{D}_1$')
ax2.scatter(np.array([1,5,9]), error[:,1],marker='o', color='r', label='$\mathrm{H}_1$')
#ax2.scatter(np.arange(1,6), error[:,2],marker='D',label='H2')
ax2.scatter(np.array([1,5,9]), error[:,3],marker='s', color='k', label='$\mathrm{H}_3$')
ax2.scatter(np.array([1,5,9]), error[:,4],marker='^', color='b', label='$\mathrm{SWA}_1$')
ax2.scatter(np.array([1,5,9]), error[:,5],marker='D', color='g', label='$\mathrm{SWA}_2$')

ax2.set_ylabel('Max Error (\%)')
ax2.legend(ncol=2)

ax2.xaxis.set_minor_locator(MultipleLocator())

ax2.tick_params(which='both', width=2)
ax2.set_xticks([1,5,9])
ax2.set_xlabel(r'$\omega_{\mathrm{SWA}_2}$')
fig2.savefig('others')

fig1, ax1 = plt.subplots()
ax1.scatter(np.array([1,5,9]), error[:,2]-4,marker='D', color='m', label='$\mathrm{H}_2$')
ax1.set_ylabel('Max Error (\%)', color='m')
ax1.tick_params(axis='y', labelcolor='m')
ax1.set_xticks([1,5,9])
ax1.yaxis.set_label_position("right")
ax1.yaxis.tick_right()

ax1.legend()
ax1.tick_params(which='both', width=2)
ax1.set_xlabel(r'$\omega_{\mathrm{SWA}_2}$')
fig1.savefig('H2')


