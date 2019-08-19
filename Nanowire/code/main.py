# -*- coding: utf-8 -*-
"""
Created on Thu Nov 8 11:25:47 2018
Updated on Tue Dec 25 2018

@author: yananliu
"""

from scipy.io import loadmat
import matplotlib.pyplot as plt

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, GlobalMaxPooling2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras.layers.normalization import BatchNormalization

# Load data 

# Play with the experiment data
# Simple visualization

    
mat = loadmat('TrainingData_4Fig3.mat')  
  
# Possible ways to play with the images
# take log for db scale
# substract all from one image
# treat as 1D image

Spec = mat['Spec_mat']
BCD = mat['BCD_mat']
HGT = mat['HGT_mat']
TCD = mat['TCD_mat']

# This extract the images in Spec
# image with 5x19 angles fixed at one single wavelength
import os
# Make directories


for wavelength_index in range(1,10):


    result_dir = 'wavelength_'+str(wavelength_index)
    os.mkdir(result_dir)

    ind_start = wavelength_index*15
    ind_end = wavelength_index*15+15
    imgs_fix_wavelength = [Spec[0,ind][ind_start:ind_end,:,:] for ind in range(Spec.shape[-1])]
    imgs_fix_wavelength = np.asarray(imgs_fix_wavelength)

    curves = imgs_fix_wavelength.reshape((3200,15*5*19))
    images_3D = imgs_fix_wavelength
    images_2D = imgs_fix_wavelength.reshape((3200,15,5*19))



    """
    plt.figure(1)
    for ind in range(5):
        plt.plot(curves[ind,:])
    plt.savefig(os.path.join(result_dir, 'sample_input1.png'))
    """

    BCD = np.swapaxes(BCD,0,1)
    HGT = np.swapaxes(HGT,0,1)
    TCD = np.swapaxes(TCD,0,1)

    # Normalize each column 
    y_max = np.asarray([np.max(BCD),np.max(HGT),np.max(TCD)])
    y_min = np.asarray([np.min(BCD),np.min(HGT),np.min(TCD)])






    def scale_y(y, y_max, y_min):
        y_scale = np.zeros(y.shape)
        for index in range(y.shape[-1]):
            y_scale[:,index] = (y[:,index]-y_min[index])/(y_max[index]-y_min[index])
        return y_scale

    # Stack into a single y vector
    y = np.stack([BCD, HGT, TCD],axis=1).reshape(3200, 3)
    y = scale_y(y, y_max, y_min)

    x = images_2D

    """
    plt.figure(2)
    plt.plot(x[1,:])
    plt.savefig(os.path.join(result_dir, 'sample_input2.png'))
    """
    # Shuffle

    # Separate Training data from Test 
    x_train = x[:2500,:]
    y_train = y[:2500,:]
    x_test = x[2500:,:]
    y_test = y[2500:,:]
    # Reshape
    input_dim = x.shape[-2:]
    output_dim = y.shape[-1]

    x_train = x_train.reshape((x_train.shape[0],1,input_dim[0],input_dim[1]))
    x_test = x_test.reshape((x_test.shape[0],1,input_dim[0],input_dim[1]))




    # Set hyper parameters

    hyper_parameters = {
    "batch_size": 2500,
    "filters": (300,50),
    "kernel_size": (1,1),
    "hidden_dims": 100,
    "epochs": 2500,
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
    model.add(Conv2D(filters[0],
                        kernel_size[0],
                        input_shape = (None, input_dim[0], input_dim[1]),
                        padding ='same',
                        activation='relu',
                        strides=1))
    model.add(BatchNormalization())
    model.add(Conv2D(filters[1],kernel_size[1], padding='same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(GlobalMaxPooling2D())
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

    csv_logger = CSVLogger(os.path.join(result_dir,'training_3D_New.log'))

    modelpath = os.path.join(result_dir, "model_3D_New.hdf5")

    checkpoint = ModelCheckpoint(modelpath, monitor='val_loss', verbose=0, 
        save_best_only=True, save_weights_only=False, mode='auto')

    lrate = LearningRateScheduler(step_decay)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                patience=200, min_lr = 1e-9)

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3000, 
        verbose=0, mode='auto')

    callbacks_list = [checkpoint, early_stop, csv_logger, lrate, reduce_lr]

    history = model.fit(x_train, y_train,
            batch_size=batch_size,
            verbose = 1,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks =  callbacks_list	
            )

    # After training, load best model and evaluate performance

    model = load_model(modelpath)
    y_ = model.predict(x_test)
    y_train_ = model.predict(x_train)




    def rescale_y(y, y_max, y_min):
        y_rescale = np.zeros(y.shape)
        for index in range(y.shape[-1]):
            y_rescale[:,index] = y[:,index]*(y_max[index]-y_min[index])+y_min[index]
        return y_rescale



    pred_rescaled = rescale_y(y_, y_max, y_min)
    train_pred_rescaled = rescale_y(y_train_, y_max, y_min)


    BCD_test = BCD[2500:]
    HGT_test = HGT[2500:]
    TCD_test = TCD[2500:]

    BCD_pred = pred_rescaled[:,0]
    HGT_pred = pred_rescaled[:,1]
    TCD_pred = pred_rescaled[:,2]

    BCD_train = BCD[:2500]
    HGT_train = HGT[:2500]
    TCD_train = TCD[:2500]


    BCD_pred_train = train_pred_rescaled[:,0]
    HGT_pred_train = train_pred_rescaled[:,1]
    TCD_pred_train = train_pred_rescaled[:,2]


    """
    plt.figure(3)
    plt.plot(BCD_test, BCD_pred, 'o')
    plt.plot(BCD_test, BCD_test,'r')
    plt.xlabel('BCD_nominal (m)')
    plt.ylabel('BCD_NN_output (m)')
    plt.title('BCD_test')
    plt.grid(b=True, which = 'both', axis='both')
    plt.savefig(os.path.join(result_dir, 'BCD_test.png'))

    plt.figure(4)
    plt.plot(HGT_test, HGT_pred, 'o')
    plt.plot(HGT_test, HGT_test,'r')
    plt.xlabel('HGT_nominal (m)')
    plt.ylabel('HGT_NN_output (m)')
    plt.title('HGT_test')
    plt.grid(b=True, which = 'both', axis='both')
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.savefig(os.path.join(result_dir, 'HGT_test.png'))

    plt.figure(5)
    plt.plot(TCD_test, TCD_pred, 'o')
    plt.plot(TCD_test, TCD_test,'r')
    plt.xlabel('TCD_nominal (m)')
    plt.ylabel('TCD_NN_output (m)')
    plt.title('TCD_test')
    plt.grid(b=True, which = 'both', axis='both')
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.savefig(os.path.join(result_dir, 'TCD_test.png'))


    plt.figure(6)
    plt.plot(BCD_train, BCD_pred_train, 'o')
    plt.plot(BCD_train, BCD_train,'r')
    plt.xlabel('BCD_nominal (m)')
    plt.ylabel('BCD_NN_output (m)')
    plt.title('BCD_train')
    plt.grid(b=True, which = 'both', axis='both')
    plt.savefig(os.path.join(result_dir, 'BCD_train.png'))


    plt.figure(7)
    plt.plot(HGT_train, HGT_pred_train, 'o')
    plt.plot(HGT_train, HGT_train,'r')
    plt.xlabel('HGT_nominal (m)')
    plt.ylabel('HGT_NN_output (m)')
    plt.title('HGT_train')
    plt.grid(b=True, which = 'both', axis='both')
    plt.ticklabel_format(style='sci', axis='both')
    plt.savefig(os.path.join(result_dir, 'HGT_train.png'))

    plt.figure(8)
    plt.plot(TCD_train, TCD_pred_train, 'o')
    plt.plot(TCD_train, TCD_train,'r')
    plt.xlabel('TCD_nominal (m)')
    plt.ylabel('TCD_NN_output (m)')
    plt.title('TCD_train')
    plt.grid(b=True, which = 'both', axis='both')
    plt.ticklabel_format(style='sci', axis='both')
    plt.savefig(os.path.join(result_dir, 'TCD_train.png'))


    # Plot training curve
    epoch = []
    train_loss = []
    test_loss = []
    import csv
    with open(os.path.join(result_dir, 'training_3D_New.log'), 'r') as csvfile:
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

    test_loss = list(map(float, test_loss))
    plt.semilogy(epoch, test_loss)
    plt.grid(b=True, which = 'both', axis='both')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title('Training Curve')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.savefig(os.path.join(result_dir,'Training_Curve.png'))
    """
    # Save .mat files
    from scipy.io import savemat
    savemat(os.path.join(result_dir,'NN_prediction_test'),{'BCD_pred_test':BCD_pred,'HGT_pred_test':HGT_pred,'TCD_pred_test':TCD_pred})

    savemat(os.path.join(result_dir,'NN_prediction_train'),{'BCD_pred_train':BCD_pred_train,'HGT_pred_train':HGT_pred_train,'TCD_pred_train':TCD_pred_train})

    savemat(os.path.join(result_dir,'data_test'),{'BCD_test':BCD_test,'HGT_test':HGT_test,'TCD_test':TCD_test})
    savemat(os.path.join(result_dir,'data_train'),{'BCD_train':BCD_train,'HGT_train':HGT_train,'TCD_train':TCD_train})



    # Inference on Experimental data
    mat_exp = loadmat('ExpData_4Fig3.mat')
    exp = mat_exp['exp_data']

    inference = model.predict(exp[ind_start:ind_end,:,:].reshape((1,1,15,5*19)))

    np.savetxt(os.path.join(result_dir,'inference_measurement.txt'),rescale_y(inference, y_max, y_min))   
