# -*- coding: utf-8 -*-
"""
Timing the inference time running repeated random tests
Created on Sun Aug 11 10:46:19 2019

@author: yliu258
"""

import time
import numpy as np
import matplotlib.pyplot as plt


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


timer = []
num_sample = np.arange(50,2500,50)
for i in np.arange(50,2500,50):
    start = time.time()
    y_pred = model.predict(x_train[0:i,:,:])
    end = time.time()
    timer.append(end-start)

timer = np.asarray(timer)
avg_timer = timer/num_sample

fig2, ax2 = plt.subplots()
ax2.plot(num_sample, avg_timer*1000, '-o')
ax2.set_ylabel('Average Time Per Sample (ms)')
labels = [item.get_text() for item in ax2.get_xticklabels()]
ax2.set_xlabel('Num of Samples')

fig, ax = plt.subplots()
ax.plot(num_sample, timer*1000, '-o')
ax.set_ylabel('Total Time (ms)')
labels = [item.get_text() for item in ax.get_xticklabels()]
ax.set_xlabel('Num of Samples')

timer_single = []
for i in np.arange(200,5000,400):
    start = time.time()
    for j in range(i):
        y_pred = model.predict(x_test[0:1,:,:])
    end = time.time()
    timer_single.append(end - start)
    
timer_single = np.asarray(timer_single)
avg_timer_single = timer_single/np.arange(200,5000,400)

fig3, ax3 = plt.subplots()
ax3.plot(np.arange(200,5000,400), avg_timer_single*1000, '-o')
ax3.set_ylabel('Avg Time Per Run (ms)')
labels = [item.get_text() for item in ax3.get_xticklabels()]
ax3.set_xlabel('Num of Repeated Runs')

    
    