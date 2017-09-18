"""
3. perform monte carlo simulation
"""

from __future__ import print_function

#%matplotlib inline
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import scipy.io as sio
# set parameters:



def read(training_ratio):

	BASE_DIR = os.getcwd()
	TRAIN_DIR = BASE_DIR + '/Noise/train/level0/'
	NOISE_DIR = BASE_DIR + '/Noise/train/level5/'
	TEST_DIR = BASE_DIR + '/Noise/test/level0/'


	files = os.listdir(TRAIN_DIR)
	x = []
	dictionary = {}
	for file in files:
	    x.append(sio.loadmat(TRAIN_DIR + file)['inputs'])
	    dictionary[file[9:13]] = files.index(file)

	length = [X.shape[0] for X in x]

	#num_training = [int(training_ratio*i) for i in length]

	num_training = [1,1,1,1]
	x_train = np.concatenate((x[0][:num_training[0]],x[1][:num_training[1]],x[2][:num_training[2]],x[3][:num_training[3]]))
	x_val = np.concatenate((x[0][num_training[0]:],x[1][num_training[1]:],x[2][num_training[2]:],x[3][num_training[3]:]))


	files = os.listdir(NOISE_DIR)
	x = []
	for file in files:
	    x.append(sio.loadmat(NOISE_DIR + file)['inputs'])
	noise = np.concatenate((x[0][:num_training[0]],x[1][:num_training[1]],x[2][:num_training[2]],x[3][:num_training[3]]))

	x_train = x_train + noise

	y = [np.repeat(i, num_training[i]) for i in range(len(length))]


	num_val = [length[i] - num_training[i] for i in range(len(length))]
	y_val = [np.repeat(i, num_val[i]) for i in range(len(length))]


	y_train = np.concatenate((y[0],y[1],y[2],y[3]))
	y_train_cat = to_categorical(y_train, num_classes = None)

	y_val = np.concatenate((y_val[0],y_val[1],y_val[2],y_val[3]))
	y_val_cat = to_categorical(y_val, num_classes = None)



	indices = np.arange(x_train.shape[0])
	np.random.shuffle(indices)
	x_train = x_train[indices]
	y_train = y_train[indices]



	indices = np.arange(x_val.shape[0])
	np.random.shuffle(indices)
	x_val = x_val[indices]
	y_val = y_val[indices]



	files = os.listdir(TEST_DIR)
	x = []
	for file in files:
	    x.append(sio.loadmat(TEST_DIR+file)['In_spec'])

	length = [X.shape[0] for X in x]

	x_test = np.concatenate((x[0],x[1],x[2],x[3]))
	y = [np.repeat(i, 100) for i in range(len(length))]

	index = [dictionary[filename[5:9]] for filename in files]
	y_test = np.concatenate((y[index[0]],y[index[1]],y[index[2]],y[index[3]]))
	y_test_cat = to_categorical(y_test, num_classes = None)



	print('Loading data...')

	print('x_train shape:', x_train.shape)
	print('x_test shape:', x_test.shape)

	return (x_train, x_val, x_test, y_train, y_val, y_test)