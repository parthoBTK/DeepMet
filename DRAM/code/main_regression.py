"""
Adding batch normalization helps a lot.
#There are parameters that are harder to train. Need more data. 
Improve training.
Use noise data.
Get confidence level.
"""

#%matplotlib inline
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import scipy.io as sio
# set parameters:
import read_data
import build_model
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.callbacks import ReduceLROnPlateau,  LearningRateScheduler
from keras import backend as K
import utils 
from keras.losses import mse

# loss function 
def _loss_mse(y_true, y_pred):
    out = K.mean(K.square(y_true - y_pred), axis= -1)
    return K.mean(out, axis=-1)

# loss function 
def _loss_modified_mse(y_true, y_pred):
    diff = (y_true - y_pred)
    sum_diff = K.square(diff[:,0]*10)
    sum_diff = sum_diff + K.square(diff[:,1])
    sum_diff = sum_diff + K.square(diff[:,2])
    sum_diff = sum_diff + K.square(diff[:,3])
    sum_diff = sum_diff + K.square(diff[:,4])
    sum_diff = sum_diff + K.square(diff[:,5])
    
    return K.mean(sum_diff)

def _loss_modified_mse(y_true, y_pred):
    diff = (y_true - y_pred)
    sum_diff = K.square(diff[:,5])
    for i in range(0,5):
        if not i == 0:
        	sum_diff = sum_diff + K.square(diff[:,i]) 
        else:
        	sum_diff = sum_diff + K.square(diff[:,i]*5) 

    return K.mean(sum_diff)

hyper_parameters = {
"batch_size": 512,
"filters": (300,50),
"kernel_size": (1,1),
"hidden_dims": 100,
"epochs": 5000,
"dropout_rate": 0.05
}

training_ratio = 1
noise_level = 0

batch_size = hyper_parameters['batch_size']
filters = hyper_parameters['filters']
kernel_size = hyper_parameters['kernel_size']
hidden_dims = hyper_parameters['hidden_dims']
epochs = hyper_parameters['epochs']


(x_train, x_val, x_test, data_train, data_val, 
	                    data_test) = read_data.read_regression("Class1", 
	                    training_ratio, noise_level)


y_train = np.apply_along_axis(utils.normalize, 0, data_train)
y_test = np.apply_along_axis(utils.normalize, 0, data_test)

if not training_ratio == 1:
    y_val = np.apply_along_axis(utils.normalize, 0, data_val)



input_dim = x_train.shape[-1]
output_dim = y_train.shape[-1]
hyper_parameters['input_dim'] = input_dim
hyper_parameters['output_dim'] = output_dim



def step_decay(epoch):
    initial_lrate = 0.001
    decay = 0.1
    epochs_decay = 1000.0
    lrate = initial_lrate * np.power(decay, np.floor((1+epoch)/epochs_decay))
    return lrate

model = build_model.regression_model(hyper_parameters)

model.compile(loss=_loss_modified_mse,
                  optimizer='adam',
                  metrics=['mse'])

#model.compile(loss='mse', optimizer='adam', metrics=['mse'])

x_train = x_train.reshape((x_train.shape[0],1,input_dim))
x_test = x_test.reshape((x_test.shape[0],1,input_dim))


if not training_ratio == 1:
	x_val = x_val.reshape((x_val.shape[0],1,input_dim))

base_dir = os.getcwd()

csv_logger = CSVLogger(os.path.join(base_dir, 'training.log'))

#filepath = "model.{epoch:02d}-{val_loss:.2f}.hdf5"
modelpath = os.path.join(base_dir, "model.hdf5")
checkpoint = ModelCheckpoint(modelpath, monitor='val_loss', verbose=1, 
	save_best_only=True, save_weights_only=True, mode='auto')


early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2000, 
	verbose=0, mode='auto')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
              patience=500, min_lr = 1e-8)

callbacks_list = [early_stop, checkpoint, csv_logger, reduce_lr]



#import time
start = time.time()

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          verbose = 0,
	      callbacks =  callbacks_list	
          )
end = time.time()
time_to_train = end - start

print(K.eval(model.optimizer.lr))

print(model.summary())
print("Num of training data: ", x_train.shape[0])

#%% Apply after training is completed
model.load_weights(modelpath)

y_test = np.apply_along_axis(utils.normalize, 0, data_test)
output_dim = y_train.shape[-1]

y_pred = model.predict(x_test)

data_pred = utils.denormalize(data_test, y_pred)

labels = ['D1(um)','H1(um)','H2(um)','H3(nm)','Theta1(deg)','Theta2(deg)', 'H4(um)','H0(um)']

relative_error = 100*np.abs(data_pred-data_test)/data_test
print(np.max(relative_error,axis=0))

np.savetxt(os.path.join(base_dir,'test_data.txt'), data_test, delimiter=',', fmt="%1.4e")
np.savetxt(os.path.join(base_dir,'model_prediction.txt'), data_pred, delimiter=',', fmt="%1.4e")
np.savetxt(os.path.join(base_dir,'relative_error.txt'), np.max(relative_error,axis=0).reshape((1,6)), delimiter=',', fmt="%1.4e")



