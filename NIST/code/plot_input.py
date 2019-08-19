# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 23:49:31 2019

@author: yliu258
"""
mat = loadmat('training.mat') 
img = mat['imag_lib']

ind = 15
i_max = img[:,:,ind].max()

plt.imshow(img[:,:,ind]/i_max, cmap='jet')
plt.xticks([0,2,4,6,8,10,12,14,16],fontsize=10)
#plt.colorbar()
plt.savefig('Input_Sample.pdf', bbox='tight')


mat = loadmat('NIST8820_Center.mat') 
img = mat['imag_lib']

ind = 1
i_max = img[:,:,ind].max()

plt.imshow(img[:,:,ind]/i_max, cmap='jet')
plt.xticks([0,2,4,6,8,10,12,14,16],fontsize=10)
#plt.colorbar()
plt.savefig('Input_Sample1.pdf', bbox='tight')