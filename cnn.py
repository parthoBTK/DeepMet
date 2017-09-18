"""
3. perform monte carlo simulation
"""


from __future__ import print_function

#%matplotlib inline
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import classification_report
import numpy as np
import csv
import os
import matplotlib.pyplot as plt


def run_cnn(x_train, x_val, x_test, y_train, y_val, y_test):
    # set parameters:
    batch_size = 128
    filters = 200
    kernel_size = 3
    hidden_dims = 250
    epochs = 50


    print('Build model...')
    model = Sequential()

    model.add(Conv1D(filters,
                     kernel_size,
                     input_dim = 401,
                     padding ='same',
                     activation='relu',
                     strides=1))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())
    model.add(Dense(hidden_dims))

    # We add a vanilla hidden layer:
    model.add(Dropout(0.5))
    model.add(Activation('relu'))



    # We add a vanilla hidden layer:
    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(4))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


    x_train = x_train.reshape((x_train.shape[0],1,401))
    x_test = x_test.reshape((400,1,401))
    x_val = x_val.reshape((x_val.shape[0],1,401))

    y_train_cat = to_categorical(y_train, num_classes = None)
    y_val_cat = to_categorical(y_val, num_classes = None)



    csv_logger = CSVLogger('training_cnn.log')

    filepath = "weights_best.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


    early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=0, mode='auto')



    callbacks_list = [early_stop, checkpoint]


    model.fit(x_train, y_train_cat,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_val, y_val_cat),
              callbacks = callbacks_list
              )


    model.load_weights(filepath)


    y_pred = model.predict_classes(x_val)
    print(classification_report(y_val,y_pred))
    print(x_val.shape[0])


    y_pred = model.predict_classes(x_test)
    print(classification_report(y_test,y_pred))
    print(x_test.shape[0])

    return