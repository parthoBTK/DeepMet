import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
def shuffle_data(data):

	indices = np.arange(data.shape[0])
	np.random.shuffle(indices)

	return data[indices]


def shuffle_pair(x, y):

	indices = np.arange(x.shape[0])
	np.random.shuffle(indices)

	return (x[indices], y[indices])

def add_noise(spectral, level_random = 0.06):
	noise = np.random.rand(len(spectral),1)
	noise_std = np.std(noise)
	Rrms = np.sqrt(np.sum(spectral*spectral)/len(spectral))  
	con_para = level_random * Rrms; 
	noise = con_para/noise_std * noise;
	noise = noise - np.mean(noise)
	return (noise+spectral.reshape(len(spectral),1)).reshape(len(spectral),)




def normalize(data):
	ma = max(data)
	mi = min(data)
	return (data-mi)/(ma-mi)


def denormalize(orig_data, data):

	ma = np.amax(orig_data, 0)
	mi = np.amin(orig_data, 0)
	denormed_data = np.zeros(data.shape)
	for i in range(data.shape[-1]):
		denormed_data[:,i] = data[:,i] * (ma[i]-mi[i]) + mi[i]
	return denormed_data


def plot_hist(y1, y2, label):
	error = 100 * (y1 - y2)/y1
	error = np.sort(error)
	plt.figure()
	plt.hist(error, bins = 10)
	xlabel(label.split('(')[0]+ ' Error (%)', fontsize=14)
	ylabel('Frequency', fontsize=14)
	plt.savefig('Error'+label.split('(')[0]+'.png')
	plt.close()
	return

def print_stat(y1, y2, label):
	error = 100 * abs(y1 - y2)/y1
	error = np.sort(error)

	n_68 = int(y1.shape[0] * .68)
	n_95 = int(y1.shape[0] * .95)
	error_68 = 	error[n_68]
	error_95 = 	error[n_95]
	a = np.array([error[0],error[-1],error_68,error_95,np.mean(error),np.sqrt(np.var(error))])
	np.savetxt('Error'+label.split('(')[0]+'.csv', a, delimiter=',',fmt = '%2f', header='min max 68 95 mean std')

	return


def plot_stat(y1, y2, label):
	error = np.abs((y1 - y2)/y1)


	n_68 = int(y1.shape[0] * .68)
	n_95 = int(y1.shape[0] * .95)


	index_68 = np.argsort(error)[n_68]
	index_95 = np.argsort(error)[n_95]

	offset_68 = abs(y1[index_68] - y2[index_68])
	offset_95 = abs(y1[index_95] - y2[index_95])
	plt.figure()
	plt.plot(y1,y1,linewidth=1, color = 'r')

	plt.plot(y1, y1+offset_95, linewidth=1, color ='b')
	plt.plot(y1, y1-offset_95, linewidth=1,color ='b')

	plt.plot(y1, y1+offset_68, linewidth=1,color ='k')
	plt.plot(y1, y1-offset_68, linewidth=1,color ='k')
	plt.scatter(y1, y2, s=2, c='b')
	#rc('axes', linewidth=1)

	fontsize = 14
	ax = gca()

	xlabel(label, fontsize=14)
	ylabel('Model prediction', fontsize=14)
	plt.savefig('y'+label.split('(')[0]+'.png')
	plt.close()

	return


def plot_save(y1, y2,label):

	os.mkdir(label.split('(')[0])
	os.chdir(label.split('(')[0])
	error = np.abs((y1 - y2)/y1)


	n_68 = int(y1.shape[0] * .68)
	n_95 = int(y1.shape[0] * .95)


	index_68 = np.argsort(error)[n_68]
	index_95 = np.argsort(error)[n_95]

	offset_68 = abs(y1[index_68] - y2[index_68])
	offset_95 = abs(y1[index_95] - y2[index_95])


	y_true = y1
	y_68_L = y1-offset_68
	y_68_R = y1+offset_68
	y_95_L = y1-offset_95
	y_95_R = y1+offset_95
	y_pred = y2

	np.savetxt('y_true.txt',y_true)
	np.savetxt('y_68_L.txt',y_68_L)
	np.savetxt('y_68_R.txt',y_68_R)
	np.savetxt('y_95_L.txt',y_95_L)
	np.savetxt('y_95_R.txt',y_95_R)
	np.savetxt('y_pred.txt',y_pred)

	return




#np.apply_along_axis(, 0, data)
