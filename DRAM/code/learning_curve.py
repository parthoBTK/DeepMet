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

training_loss = []
val_loss = []
epoch = []
for ind in [500,1000,1500,2000,2500]:
    hist = np.loadtxt(str(ind)+'/training.log', delimiter=',', usecols=range(5), skiprows = 1)
    training_loss.append(hist[:,1])
    val_loss.append(hist[:,3])
    epoch.append(hist[:,0])

training_loss = np.asarray(training_loss) 
val_loss = np.asarray(val_loss)
epoch = np.asarray(epoch)

fig, ax = plt.subplots()

ax.semilogy(epoch[0],val_loss[0], '-',label='500')
ax.semilogy(epoch[1],val_loss[1], '-.',label='1000')
ax.semilogy(epoch[2],val_loss[2], ':', label='1500')
ax.semilogy(epoch[3],val_loss[3], '-', label='2000')
ax.semilogy(epoch[4],val_loss[4], '-.', label='2500')

ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Loss (MSE)')
ax.legend(loc='upper right')

fig.savefig('training_curve')


#%% Plot mean error 
error = []
for ind in [500,1000,1500,2000,2500]:
    data_pred = np.loadtxt(str(ind)+'/model_prediction.txt', delimiter=',')
    data_test = np.loadtxt(str(ind)+'/test_data.txt', delimiter=',')
    relative_error = 100*np.abs(data_pred-data_test)/data_test
    error.append(np.max(relative_error,axis=0))
    
error = np.asarray(error)

fig2, ax2 = plt.subplots()
ax2.scatter(np.arange(5), error[:,0],marker='*',label='D1')
ax2.scatter(np.arange(5), error[:,1],marker='o', label='H1')
ax2.scatter(np.arange(5), error[:,3],marker='s',label='H3')
ax2.scatter(np.arange(5), error[:,4],marker='^',label='Theta1')
ax2.scatter(np.arange(5), error[:,5],marker='D',label='Theta2')

ax2.set_ylabel('Max Error (\%)')
ax2.legend()

labels = [item.get_text() for item in ax2.get_xticklabels()]

ax2.set_xticklabels(['', 500,1000,1500,2000,2500])
ax2.set_xlabel('Num of Training Data')

#ax3.legend()

fig3, ax3 = plt.subplots()
ax3.scatter(np.arange(1,6),error[:,2], c='m', marker='X', label='H2')
ax3.yaxis.set_label_position("right")
ax3.yaxis.tick_right()
ax3.set_ylabel('Max Error (\%)',color='m')
ax3.tick_params(axis='y', labelcolor='m')
ax3.legend(loc='upper right')
ax3.set_xticklabels(['', 500,1000,1500,2000,2500])
ax3.set_xlabel('Num of Training Data')


