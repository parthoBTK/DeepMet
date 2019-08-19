
from __future__ import print_function

#%matplotlib inline
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import scipy.io as sio
import utils




def read_classification(training_ratio, noise_level):

	BASE_DIR = os.getcwd()
	TRAIN_DIR = BASE_DIR + '/train/'
	#NOISE_DIR = BASE_DIR + '/Noise/train/level5/'
	TEST_DIR = BASE_DIR + '/test/'


	files =  os.listdir(TRAIN_DIR)
	x = []
	dictionary = {}
	for file in files:
	    x.append(sio.loadmat(TRAIN_DIR + file)[file[:6]+"_spec"])
	    dictionary[file[:6]] = files.index(file)

	length = [X.shape[0] for X in x]

	num_training = [int(training_ratio*i) for i in length]


	x = [utils.shuffle_data(data) for data in x]

	x_train = np.concatenate((x[0][:num_training[0]],x[1][:num_training[1]],x[2][:num_training[2]],x[3][:num_training[3]]))
	x_train = np.apply_along_axis(utils.add_noise, 0, x_train, noise_level)

	x_val = np.zeros(1)
	if not training_ratio == 1:
		x_val = np.concatenate((x[0][num_training[0]:],x[1][num_training[1]:],x[2][num_training[2]:],x[3][num_training[3]:]))
		x_val = np.apply_along_axis(utils.add_noise, 0, x_val, noise_level)

	

	y = [np.repeat(i, num_training[i]) for i in range(len(length))]

	y_train = np.concatenate((y[0],y[1],y[2],y[3]))
	(x_train, y_train) = utils.shuffle_pair(x_train, y_train)
	
	y_val = np.zeros(1)

	if not training_ratio == 1:
		num_val = [length[i] - num_training[i] for i in range(len(length))]
		y_val = [np.repeat(i, num_val[i]) for i in range(len(length))]
		y_val = np.concatenate((y_val[0],y_val[1],y_val[2],y_val[3]))
		(x_val, y_val) = utils.shuffle_pair(x_val, y_val)



	files = os.listdir(TEST_DIR)
	x = []
	for file in files:
	    x.append(sio.loadmat(TEST_DIR+file)[file[:6]+"_spec"])

	length = [X.shape[0] for X in x]

	x_test = np.concatenate((x[0],x[1],x[2],x[3]))
	y = [np.repeat(i, length[-1]) for i in range(len(length))]

	index = [dictionary[filename[:6]] for filename in files]
	y_test = np.concatenate((y[index[0]],y[index[1]],y[index[2]],y[index[3]]))



	print('Loading data...')

	print('x_train shape:', x_train.shape)
	print('x_val shape:', x_val.shape)
	print('x_test shape:', x_test.shape)


	return (x_train, x_val, x_test, y_train, y_val, y_test)


def read_regression(class_name, training_ratio, noise_level):
	BASE_DIR = os.getcwd()
	TRAIN_DIR = BASE_DIR + '/train/'
	#NOISE_DIR = BASE_DIR + '/Noise/train/level5/'
	TEST_DIR = BASE_DIR + '/test/'

	file_dir = os.path.join(TRAIN_DIR, class_name)
	data = sio.loadmat(file_dir)
	x_train = data[class_name + '_spec']
	y_train = data[class_name + '_para']

	ADDED_DIR = BASE_DIR + '/Added_Data/'
	file_dir = os.path.join(ADDED_DIR, (class_name + '_2nd'))
	data = sio.loadmat(file_dir)
	x_added = data[class_name + '_spec']
	y_added = data[class_name + '_para']
	(x_added, y_added) = utils.shuffle_pair(x_added, y_added)

	x_train = np.concatenate((x_train, x_added), axis=0)
	x_train = np.apply_along_axis(utils.add_noise, 0, x_train, noise_level)

	y_train = np.concatenate((y_train, y_added), axis=0)

	(x_train, y_train) = utils.shuffle_pair(x_train, y_train)

	file_dir = os.path.join(TEST_DIR, class_name + "_Test")
	data = sio.loadmat(file_dir)
	x_test = data[class_name + '_spec']
	y_test = data[class_name + '_para']
	(x_test, y_test) = utils.shuffle_pair(x_test, y_test)


	num_training = int(y_train.shape[0] * training_ratio)


	y_val = y_train[num_training:]
	x_val = x_train[num_training:]
	y_train = y_train[:num_training]
	x_train = x_train[:num_training]


	return (x_train, x_val, x_test , y_train, y_val, y_test)

def sub_group_wavelength(data, num_wavelength):
    num_data = data.shape[0]
    len_spectrum = data.shape[1]
    num_group = int(len_spectrum/15/num_wavelength)
    data = data.reshape((num_data, 15, int(len_spectrum/15)))
    reshaped_data = []
    for ind in range(num_group):
        reshaped_data.append(data[:,:,
                        num_wavelength*ind:num_wavelength*ind+num_wavelength].reshape((num_data, 
                        num_wavelength*15)))
    return np.asarray(reshaped_data)
    
    
