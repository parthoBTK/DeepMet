
#%matplotlib inline
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import scipy.io as sio
from sklearn.metrics import classification_report
from keras.models import load_model
import read_data
import utils


training_ratio = 1

(x_train, x_val, x_test, data_train, data_val, data_test) = read_data.read_regression("Class1", training_ratio, 0)


model.load_weights(modelpath)

input_dim = x_train.shape[-1]

x_train = x_train.reshape((x_train.shape[0],1,x_train.shape[-1]))
x_test = x_test.reshape((x_test.shape[0],1,x_test.shape[-1]))
if not training_ratio == 1:
	x_val = x_val.reshape((x_val.shape[0],1,input_dim))
	y_val = np.apply_along_axis(utils.normalize, 0, data_val)



y_train = np.apply_along_axis(utils.normalize, 0, data_train)
y_test = np.apply_along_axis(utils.normalize, 0, data_test)

output_dim = y_train.shape[-1]
start = time.time()
y_pred = model.predict(x_test)
end = time.time()
end - start

data_pred = utils.denormalize(data_test, y_pred)



labels = ['D1(um)','H1(um)','H2(um)','H3(nm)','Theta1(deg)','Theta2(deg)', 'H4(um)','H0(um)']


relative_error = 100*np.abs(data_pred-data_test)/data_test
print(np.mean(relative_error,axis=0))

np.savetxt('test_data.txt', data_test, delimiter=',', fmt="%1.4e")
np.savetxt('model_prediction.txt', data_pred, delimiter=',', fmt="%1.4e")
np.savetxt('relative_error.txt', np.mean(relative_error,axis=0).reshape((1,6)), delimiter=',', fmt="%1.4e")

hist = np.loadtxt('training.log', delimiter=',', usecols=range(5), skiprows = 1)
plt.semilogy(hist[:,0],hist[:,1],hist[:,0],hist[:,3])
plt.savefig('training_curve.png')
