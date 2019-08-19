# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:25:47 2018

@author: yananliu
"""

from scipy.io import loadmat
import matplotlib.pyplot as plt

import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, LearningRateScheduler
from sklearn.metrics import classification_report
from keras.layers.normalization import BatchNormalization

from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

# Load data 
"""
mat = loadmat('2um_Lines_Lib.mat')

SWA_lib = mat['SWA_lib']
HGT_lib = mat['HGT_lib']
BCD_lib = mat['BCD_lib']
curve = mat['fit_lib']
"""
mat = loadmat('NIST8820_Center.mat')

# Possible ways to play with the images
# take log for db scale
# substract all from one image
# treat as 1D image

imgs = mat['imag_lib']
imgs = np.swapaxes(imgs,0,2)

curves = imgs.reshape((1350,17*17))/1e6

H = np.swapaxes(mat['H_mat'],0,1)
L = np.swapaxes(mat['L_mat'],0,1)
R = np.swapaxes(mat['R_mat'],0,1)
W = np.swapaxes(mat['W_mat'],0,1)

# Normalize each column 
y_max = np.asarray([np.max(H),np.max(L),np.max(R),np.max(W)])
y_min = np.asarray([np.min(H),np.min(L),np.min(R),np.min(W)])

"""
y_max['H'] = np.max(H)
y_max['L'] = np.max(L)
y_max['R'] = np.max(R)
y_max['W'] = np.max(W)

y_min = {}
y_min['H'] = np.min(H)
y_min['L'] = np.min(L)
y_min['R'] = np.min(R)
y_min['W'] = np.min(W)
"""


def scale_y(y, y_max, y_min):
    y_scale = np.zeros(y.shape)
    for index in range(y.shape[-1]):
        y_scale[:,index] = (y[:,index]-y_min[index])/(y_max[index]-y_min[index])
    return y_scale

# Stack into a single y vector
y = np.stack([H, L, R, W],axis=1).reshape(1350,4)
y = scale_y(y, y_max, y_min)

x = curves
plt.plot(x[1,:])
# Shuffle

# Separate Training data from Test 
x_train = x[:1150,:]
y_train = y[:1150,:]
x_test = x[1150:,:]
y_test = y[1150:,:]
# Reshape
input_dim = x.shape[-1]
output_dim = y.shape[-1]

x_train = x_train.reshape((x_train.shape[0],1,input_dim))
x_test = x_test.reshape((x_test.shape[0],1,input_dim))




# Set hyper parameters

hyper_parameters = {
"batch_size": 4096,
"filters": (300,50),
"kernel_size": (1,1),
"hidden_dims": 100,
"epochs": 10000,
"dropout_rate": 0.05
}


hyper_parameters['input_dim'] = input_dim
hyper_parameters['output_dim'] = output_dim


batch_size = hyper_parameters['batch_size']
filters = hyper_parameters['filters']
kernel_size = hyper_parameters['kernel_size']
hidden_dims = hyper_parameters['hidden_dims']
epochs = hyper_parameters['epochs']
dropout_rate = hyper_parameters['dropout_rate']
input_dim = hyper_parameters['input_dim']
output_dim = hyper_parameters['output_dim']


# Build Model
model = Sequential()
model.add(Conv1D(filters[0],
	                 kernel_size[0],
	                 input_dim = input_dim,
	                 padding ='same',
	                 activation='relu',
	                 strides=1))
model.add(BatchNormalization())
model.add(Conv1D(filters[1],kernel_size[1], padding='same', activation = 'relu'))
model.add(BatchNormalization())
model.add(GlobalMaxPooling1D())
#model.add(Dropout(dropout_rate))

# we use max pooling:
model.add(Dense(hidden_dims, activation = 'relu'))
model.add(BatchNormalization())
#model.add(Dropout(dropout_rate))


model.add(Dense(50))
model.add(Activation('relu'))
model.add(BatchNormalization())
#model.add(Dropout(dropout_rate))
model.add(Dense(output_dim))
model.add(Activation('sigmoid'))

model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mse'])

def step_decay(epoch):
    initial_lrate = 0.001
    decay = 0.1
    epochs_decay = 1000.0
    lrate = initial_lrate * np.power(decay, np.floor((1+epoch)/epochs_decay))
    return lrate

csv_logger = CSVLogger('training_3D.log')

filepath = "model_3D.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
	save_best_only=True, save_weights_only=False, mode='auto')

lrate = LearningRateScheduler(step_decay)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
              patience=200, min_lr = 1e-9)

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3000, 
	verbose=0, mode='auto')

callbacks_list = [checkpoint, early_stop, csv_logger, lrate, reduce_lr]

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          verbose = 1,
	      callbacks =  callbacks_list	
          )

model = load_model('model_3D.hdf5')
y_ = model.predict(x_test)
y_train_ = model.predict(x_train)

def rescale(y_, small, large):
    return y_*(large-small) + small



def rescale_y(y, y_max, y_min):
    y_rescale = np.zeros(y.shape)
    for index in range(y.shape[-1]):
        y_rescale[:,index] = y[:,index]*(y_max[index]-y_min[index])+y_min[index]
    return y_rescale



pred_rescaled = rescale_y(y_, y_max, y_min)
train_pred_rescaled = rescale_y(y_train_, y_max, y_min)


H_test = H[1150:]
L_test = L[1150:]
R_test = R[1150:]
W_test = W[1150:]

H_pred = pred_rescaled[:,0]
L_pred = pred_rescaled[:,1]
R_pred = pred_rescaled[:,2]
W_pred = pred_rescaled[:,3]


H_train = H[:1150]
L_train = L[:1150]
R_train = R[:1150]
W_train = W[:1150]

H_pred_train = train_pred_rescaled[:,0]
L_pred_train = train_pred_rescaled[:,1]
R_pred_train = train_pred_rescaled[:,2]
W_pred_train = train_pred_rescaled[:,3]



plt.figure(1)
plt.plot(H_test, H_pred, 'o')
plt.plot(H_test, H_test,'r')
plt.xlabel('H_nominal (m)')
plt.ylabel('H_NN_output (m)')
plt.title('H_test')
plt.grid(b=True, which = 'both', axis='both')


plt.figure(2)
plt.plot(L_test, L_pred, 'o')
plt.plot(L_test, L_test,'r')
plt.xlabel('L_nominal (m)')
plt.ylabel('L_NN_output (m)')
plt.title('L_test')
plt.grid(b=True, which = 'both', axis='both')
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

plt.figure(3)
plt.plot(R_test, R_pred, 'o')
plt.plot(R_test, R_test,'r')
plt.xlabel('R_nominal (m)')
plt.ylabel('R_NN_output (m)')
plt.title('R_test')
plt.grid(b=True, which = 'both', axis='both')
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

plt.figure(4)
plt.plot(W_test, W_pred, 'o')
plt.plot(W_test, W_test,'r')
plt.xlabel('W_nominal (m)')
plt.ylabel('W_NN_output (m)')
plt.title('W_test')
plt.grid(b=True, which = 'both', axis='both')
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))


plt.figure(5)
plt.plot(H_train, H_pred_train, 'o')
plt.plot(H_train, H_train,'r')
plt.xlabel('H_nominal (m)')
plt.ylabel('H_NN_output (m)')
plt.title('H_train')
plt.grid(b=True, which = 'both', axis='both')


plt.figure(6)
plt.plot(L_train, L_pred_train, 'o')
plt.plot(L_train, L_train,'r')
plt.xlabel('L_nominal (m)')
plt.ylabel('L_NN_output (m)')
plt.title('L_train')
plt.grid(b=True, which = 'both', axis='both')
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

plt.figure(7)
plt.plot(R_train, R_pred_train, 'o')
plt.plot(R_train, R_train,'r')
plt.xlabel('R_nominal (m)')
plt.ylabel('R_NN_output (m)')
plt.title('R_train')
plt.grid(b=True, which = 'both', axis='both')
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

plt.figure(8)
plt.plot(W_train, W_pred_train, 'o')
plt.plot(W_train, W_train,'r')
plt.xlabel('W_nominal (m)')
plt.ylabel('W_NN_output (m)')
plt.title('W_train')
plt.grid(b=True, which = 'both', axis='both')
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))


# Plot training curve
epoch = []
train_loss = []
test_loss = []
import csv
with open('training_3D.log', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)
    for row in reader:
        epoch.append(row[0])
        train_loss.append(row[1])
        test_loss.append(row[3])
        
epoch = list(map(int, epoch))
train_loss = list(map(float, train_loss))
plt.figure(9)
plt.semilogy(epoch, train_loss)
plt.grid(b=True, which = 'both', axis='both')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training Error')

plt.figure(10)
test_loss = list(map(float, test_loss))
plt.semilogy(epoch, test_loss)
plt.grid(b=True, which = 'both', axis='both')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Test Error')

# Save .mat files
from scipy.io import savemat
savemat('NN_prediction_test',{'H_pred_test':H_pred,'L_pred_test':L_pred,'R_pred_test':R_pred,'W_pred_test':W_pred})
savemat('NN_prediction_train',{'H_pred_train':H_pred_train,'L_pred_train':L_pred_train,'R_pred_train':R_pred_train,'W_pred_train':W_pred_train})
savemat('data_test',{'H_test':H_test,'L_test':L_test,'R_test':R_test,'W_test':W_test})
savemat('data_train',{'H_train':H_train,'L_train':L_train,'R_train':R_train,'W_train':W_train})

"""
# Inference on Experimental data
measured_data = loadmat('exprimental_data_NIST.mat')['Exp_data']
measured_data = measured_data.transpose()
measured_data = measured_data.reshape((measured_data.shape[0],1,input_dim))

measured_pred = model.predict(measured_data)
measured_pred_rescaled = rescale(measured_pred)

from scipy.io import savemat
savemat('experimental_pred.mat', {'experimental_pred':measured_pred_rescaled})
savemat('sim_test_pred.mat', {'sim_test_pred':pred_rescaled})
savemat('sim_train_pred.mat', {'sim_train_pred':train_pred_rescaled})

from scipy.stats import mode
print("Mode for SWA is: ")
print(mode(measured_pred_rescaled[0,:]))
print("Mode for HGT is: ")
print(mode(measured_pred_rescaled[1,:]))
print("Mode for BCD is: ")
print(mode(measured_pred_rescaled[2,:]))

avg = np.mean(measured_pred_rescaled, axis=1)
print("Mean for SWA is: " + '{:.2e}'.format(avg[0]) + '\n') 
print("Mean for HGT is: " +  '{:.4E}'.format(avg[1]) + '\n')
print("Mean for BCD is: " +  '{:.4e}'.format(avg[2]) + '\n' )


std = np.std(measured_pred_rescaled, axis=1)
print("Standard deviation for SWA is: " + '{:.2e}'.format(std[0]) + '\n') 
print("Standard deviation for HGT is: " +  '{:.4E}'.format(std[1]) + '\n')
print("Standard deviation for BCD is: " +  '{:.4e}'.format(std[2]) + '\n' )



plt.figure(5)
plt.ticklabel_format(style='plain')
plt.hist(measured_pred_rescaled[0,:])

plt.figure(6)
plt.hist(measured_pred_rescaled[1,:])
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.figure(7)
plt.hist(measured_pred_rescaled[2,:])
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
"""

# What about calculate avg of the curve? And use avg_ed curve to predict?