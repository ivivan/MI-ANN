import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from math import sqrt
import seaborn as sns
from scipy import stats
from math import sqrt,fabs
from keras import backend as K
import os
import h5py
import argparse
from keras.models import Sequential
from keras.layers import Dense, Dropout,RepeatVector
from keras.layers import LSTM, TimeDistributed, Masking
from keras.callbacks import Callback,ModelCheckpoint
from keras.models import load_model
from keras import regularizers
from keras.optimizers import SGD

import tensorflow as tf
import tensorboard


# tf.logging.set_verbosity(tf.logging.INFO)
# np.random.seed(1337)
# Prepare data

def rnn_data(data, time_steps, labels=False):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]] #Data frame for input with 2 timesteps
        -> labels == True [3, 4, 5] # labels for predicting the next timestep
    """
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:
            try:
                rnn_df.append(data.iloc[i + time_steps].as_matrix())
            except AttributeError:
                rnn_df.append(data.iloc[i + time_steps])
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

    return np.array(rnn_df, dtype=np.float64)


def change_predictive_interval(train,label,predictiveinterval):
    label = label[predictiveinterval-1:]
    train = train[:len(label)]

    return train,label



def load_csvdata(rawdata, time_steps,mode,scalardic):
    data = rawdata
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    train_x = rnn_data(data['DO_mg'], time_steps, labels=False)
    train_y = rnn_data(data['DO_mg'], time_steps, labels=True)

    train_x_two = rnn_data(data['EC_uScm'], time_steps, labels=False)
    train_y_two = rnn_data(data['EC_uScm'], time_steps, labels=True)

    train_x_three = rnn_data(data['Temp_degC'], time_steps, labels=False)
    train_y_three = rnn_data(data['Temp_degC'], time_steps, labels=True)

    train_x_four = rnn_data(data['pH'], time_steps, labels=False)
    train_y_four = rnn_data(data['pH'], time_steps, labels=True)

    train_x_five = rnn_data(data['Chloraphylla_ugL'], time_steps, labels=False)
    train_y_five = rnn_data(data['Chloraphylla_ugL'], time_steps, labels=True)

    train_x = np.squeeze(train_x)
    train_x_two = np.squeeze(train_x_two)
    train_x_three = np.squeeze(train_x_three)
    train_x_four = np.squeeze(train_x_four)
    train_x_five = np.squeeze(train_x_five)

    # Scale data (training set) to 0 mean and unit standard deviation.
    if(mode=='train'):
        scaler_do = scalardic['scaler_one']
        scaler_ec = scalardic['scaler_two']
        scaler_temp = scalardic['scaler_three']
        scaler_ph = scalardic['scaler_four']
        scaler_chlo = scalardic['scaler_five']

        train_x = scaler_do.fit_transform(train_x)
        train_x_two = scaler_ec.fit_transform(train_x_two)
        train_x_three = scaler_temp.fit_transform(train_x_three)
        train_x_four =  scaler_ph.fit_transform(train_x_four)
        train_x_five = scaler_chlo.fit_transform(train_x_five)

    elif (mode=='test'):
        scaler_do = scalardic['scaler_one']
        scaler_ec = scalardic['scaler_two']
        scaler_temp = scalardic['scaler_three']
        scaler_ph = scalardic['scaler_four']
        scaler_chlo = scalardic['scaler_five']

        train_x = scaler_do.transform(train_x)
        train_x_two = scaler_ec.transform(train_x_two)
        train_x_three = scaler_temp.transform(train_x_three)
        train_x_four =  scaler_ph.transform(train_x_four)
        train_x_five = scaler_chlo.transform(train_x_five)

    all_train = np.stack((train_x, train_x_two, train_x_three,train_x_four,train_x_five), axis=-1)
    all_train = all_train.reshape(-1,time_steps*5)

    return dict(train=all_train,scalerone=scaler_do,scalertwo=scaler_ec,scalerthree=scaler_temp,scalerfour=scaler_ph,scalerfive=scaler_chlo), dict(trainyone=train_y,trainytwo=train_y_two,trainythree=train_y_three)


def data_together(filepath):

    csvs = []
    dfs = []

    for subdir, dirs, files in os.walk(filepath):
        for file in files:
            # print os.path.join(subdir, file)
            filepath = subdir + os.sep + file
            if filepath.endswith(".hdf5"):
                csvs.append(filepath)

    return csvs


def generate_data(filepath, num_sample, timestamp, start, mode, scalardic):
    """
    :param filepath: data set for the model
    :param start: start row for training set, for training, start=0
    :param num_sample: how many samples used for training set, in this case, 2928 samples from 1st Oct-30th Nov, two month
    :param timestamp: timestamp used for LSTM
    :return: training set, train_x and train_y
    """
    dataset = pd.read_csv(filepath)

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for c in [c for c in dataset.columns if dataset[c].dtype in numerics]:
        dataset[c] = dataset[c].abs()

    dataset = dataset.iloc[start:start + num_sample, :]  # get first num_sample rows for training set, with all columns
    dataset['TIMESTAMP'] = pd.to_datetime(dataset['TIMESTAMP'], dayfirst=True)



    set_x, set_y = load_csvdata(dataset, timestamp, mode, scalardic)
    return set_x, set_y


def getdate_index(filepath, start, num_predict):
    """
    :param filepath: same dataset file
    :param start: start now no. for prediction
    :param num_predict: how many predictions
    :return: the x axis datatime index for prediciton drawing
    """
    dataset = pd.read_csv(filepath)
    dataset = dataset.iloc[start:start + num_predict, :]
    dataset['TIMESTAMP'] = pd.to_datetime(dataset['TIMESTAMP'], dayfirst=True)

    return dataset['TIMESTAMP']



def rsquare(y_true,y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))

if __name__ == '__main__':

    # Parameters
    model_params = {
        'TIMESTEPS': 12,
        'N_FEATURES':5,
        'train_no': 2452,
        'test_no':496
    }


# Scale x data (training set) to 0 mean and unit standard deviation.
    scaler_do = preprocessing.StandardScaler()
    scaler_ec = preprocessing.StandardScaler()
    scaler_temp = preprocessing.StandardScaler()
    scaler_ph = preprocessing.StandardScaler()
    scaler_chlo = preprocessing.StandardScaler()

    scaler_dic = {
        'scaler_one': scaler_do,
        'scaler_two': scaler_ec,
        'scaler_three': scaler_temp,
        'scaler_four': scaler_ph,
        'scaler_five': scaler_chlo
    }

# datafile

    filepath = './dry_season-90min.csv'
    x, y = generate_data(filepath, model_params['train_no'], model_params['TIMESTEPS'], 0, 'train', scaler_dic)

    scaler_dic['scaler_one'] = x['scalerone']
    scaler_dic['scaler_two'] = x['scalertwo']
    scaler_dic['scaler_three'] = x['scalerthree']
    scaler_dic['scaler_four'] = x['scalerfour']
    scaler_dic['scaler_five'] = x['scalerfive']

# Training set, three train y for multiple tasks training
    x_train = x['train']
    y_train_do = y['trainyone']
    y_train_ec = y['trainytwo']
    y_train_temp = y['trainythree']

    x_t, y_t = generate_data(filepath, model_params['test_no']+12, model_params['TIMESTEPS'], model_params['train_no']-12, 'test', scaler_dic)  # testing set for 240 prediction (5 days) 732 372
# Testing set, three test y for multiple tasks testing
    x_test = x_t['train']
    y_test_do = y_t['trainyone']
    y_test_ec = y_t['trainytwo']
    y_test_temp = y_t['trainythree']

# Scale y data to 0 mean and unit standard deviation
    scaler_do_y = preprocessing.StandardScaler()
    scaler_ec_y = preprocessing.StandardScaler()
    scaler_temp_y = preprocessing.StandardScaler()

    y_train_do = y_train_do.reshape(-1, 1)
    y_train_ec = y_train_ec.reshape(-1, 1)
    y_train_temp = y_train_temp.reshape(-1, 1)


    y_train_do = scaler_do_y.fit_transform(y_train_do)
    y_train_ec = scaler_ec_y.fit_transform(y_train_ec)
    y_train_temp = scaler_temp_y.fit_transform(y_train_temp)

    y_test_do = y_test_do.reshape(-1, 1)
    y_test_ec = y_test_ec.reshape(-1,1)
    y_test_temp = y_test_temp.reshape(-1,1)

    y_test_do = scaler_do_y.transform(y_test_do)
    y_test_ec = scaler_ec_y.transform(y_test_ec)
    y_test_temp = scaler_temp_y.transform(y_test_temp)


    x_train = x_train.reshape((x_train.shape[0], model_params['TIMESTEPS']* model_params['N_FEATURES']))
    x_test = x_test.reshape((x_test.shape[0],model_params['TIMESTEPS']* model_params['N_FEATURES']))

##
# Model
##

    model = Sequential()
    model.add(Dense(3, input_dim=model_params['TIMESTEPS']* model_params['N_FEATURES'], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='relu'))
# model.add(Dropout(0.2))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1, activation=None))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer='sgd',metrics=['mse','mae',rsquare])


    trained_model = []

    trained_each = os.path.abspath(os.path.join('./', str(3)))

    print(trained_each)


    trained_model = data_together(trained_each)

    for i in trained_model:
        model = load_model(i, custom_objects={'rsquare': rsquare})
        predictions = model.predict(x_test)

        y_predicted = np.array(list(scaler_do_y.inverse_transform(p) for p in predictions))

        y_predicted = y_predicted.reshape(np.array(y_test_do).shape)

        print("---------")
        r2 = r2_score(scaler_do_y.inverse_transform(y_test_do), y_predicted)
        print("File Name{0}-R2 (sklearn):{1:f}".format(i,r2))