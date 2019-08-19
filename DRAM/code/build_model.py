from __future__ import print_function

#%matplotlib inline
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import scipy.io as sio
# set parameters:
import read_data
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, LearningRateScheduler
from sklearn.metrics import classification_report
from keras.layers.normalization import BatchNormalization


def classification_model(hyper_parameters):

	# Set parameters
	batch_size = hyper_parameters['batch_size']
	filters = hyper_parameters['filters']
	kernel_size = hyper_parameters['kernel_size']
	hidden_dims = hyper_parameters['hidden_dims']
	epochs = hyper_parameters['epochs']
	dropout_rate = hyper_parameters['dropout_rate']
	input_dim = hyper_parameters['input_dim']

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
	model.add(Dropout(dropout_rate))

	# we use max pooling:
	model.add(Dense(hidden_dims, activation = 'relu'))
	model.add(BatchNormalization())

	# We add a vanilla hidden layer:
	model.add(Dropout(dropout_rate))

	# We add a vanilla hidden layer:
	# We project onto a single unit output layer, and squash it with a sigmoid:
	model.add(Dense(4))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])
	return model


def regression_model(hyper_parameters):

	# Set parameters
	batch_size = hyper_parameters['batch_size']
	filters = hyper_parameters['filters']
	kernel_size = hyper_parameters['kernel_size']
	hidden_dims = hyper_parameters['hidden_dims']
	epochs = hyper_parameters['epochs']
	dropout_rate = hyper_parameters['dropout_rate']
	input_dim = hyper_parameters['input_dim']
	output_dim = hyper_parameters['output_dim']


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
	model.add(Dropout(dropout_rate))

	# we use max pooling:
	model.add(Dense(hidden_dims, activation = 'relu'))
	model.add(BatchNormalization())
	model.add(Dropout(dropout_rate))

	# We add a vanilla hidden layer:
	# We project onto a single unit output layer, and squash it with a sigmoid:
	model.add(Dense(50))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(dropout_rate))
	
	model.add(Dense(output_dim))
	model.add(Activation('sigmoid'))




	
	return model

