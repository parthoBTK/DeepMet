#!/usr/bin/python
import read_data
import cnn

training_ratio = .001

(x_train,x_val,x_test,y_train,y_val,y_test) = read_data.read(training_ratio)





hyper_parameters = {
    "batch_size": 128,
    "filters": 200,
    "kernel_size": 3,
    "hidden_dims": 250,
    "epochs": 30
    }



cnn.run_cnn(x_train, x_val, x_test, y_train, y_val, y_test)