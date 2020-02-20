from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import pywt
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import preprocessing
from math import sqrt
from data_process import data_io_improve as dataio
from forfirstconf.LSTM_Support import lstm_model

tf.logging.set_verbosity(tf.logging.INFO)


# Prepare data

def generate_data(filepath, num_sample, timestamp, start, mode, scalardic):
    """
    :param filepath: data set for the model
    :param start: start row for training set, for training, start=0
    :param num_sample: how many samples used for training set, in this case, 2928 samples from 1st Oct-30th Nov, two month
    :param timestamp: timestamp used for LSTM
    :return: training set, train_x and train_y
    """
    dataset = pd.read_csv(filepath)
    dataset = dataset.iloc[start:start + num_sample, :]  # get first num_sample rows for training set, with all columns
    dataset['TIMESTAMP'] = pd.to_datetime(dataset['TIMESTAMP'], dayfirst=True)

    # wavelet
    ntu_list = dataset['Turbidity_NTU'].tolist()

    set_x, set_y = dataio.load_csvdata(dataset, timestamp, mode, scalardic)
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


# Parameters
model_params = {
    'TIMESTEPS': 3,
    'N_FEATURES':3,
    'RNN_LAYERS': [{'num_units': 400}],
    # 'RNN_LAYERS': [{'num_units': 60, 'keep_prob': 0.75},{'num_units': 120, 'keep_prob': 0.75},{'num_units': 60, 'keep_prob': 0.75}],
    'DENSE_LAYERS': None,
    'TRAINING_STEPS': 15,
    'PRINT_STEPS': 50,
    'BATCH_SIZE': 100
}


# Scale x data (training set) to 0 mean and unit standard deviation.
scaler_ntu = preprocessing.StandardScaler()
scaler_ec = preprocessing.StandardScaler()
scaler_temp = preprocessing.StandardScaler()

scaler_dic = {
    'scaler_one': scaler_ntu,
    'scaler_two': scaler_ec,
    'scaler_three': scaler_temp
}

# datafile
filepath = r'C:\Users\ZHA244\Coding\QLD\baffle_creek\baffle-creek-buoy-quality-2013-multiple.csv'
x, y = generate_data(filepath, 2928,model_params['TIMESTEPS'], 0, 'train', scaler_dic)

scaler_dic['scaler_one'] = x['scalerone']
scaler_dic['scaler_two'] = x['scalertwo']
scaler_dic['scaler_three'] = x['scalerthree']

# Training set, three train y for multiple tasks training
x_train = x['train']
y_train_ntu = y['trainyone']
y_train_ec = y['trainytwo']
y_train_temp = y['trainythree']

x_t, y_t = generate_data(filepath, 243, model_params['TIMESTEPS'], 2925, 'test', scaler_dic)  # testing set for 240 prediction (5 days)
# Testing set, three test y for multiple tasks testing
x_test = x_t['train']
y_test_ntu = y_t['trainyone']
y_test_ec = y_t['trainytwo']
y_test_temp = y_t['trainythree']

# Scale y data to 0 mean and unit standard deviation
scaler_ntu_y = preprocessing.StandardScaler()
scaler_ec_y = preprocessing.StandardScaler()
scaler_temp_y = preprocessing.StandardScaler()

y_train_ntu = y_train_ntu.reshape(-1, 1)
y_train_ec = y_train_ec.reshape(-1, 1)
y_train_temp = y_train_temp.reshape(-1, 1)


y_train_ntu = scaler_ntu_y.fit_transform(y_train_ntu)
y_train_ec = scaler_ec_y.fit_transform(y_train_ec)
y_train_temp = scaler_temp_y.fit_transform(y_train_temp)

y_test_ntu = y_test_ntu.reshape(-1,1)
y_test_ec = y_test_ec.reshape(-1,1)
y_test_temp = y_test_temp.reshape(-1,1)

y_test_ntu = scaler_ntu_y.transform(y_test_ntu)
y_test_ec = scaler_ec_y.transform(y_test_ec)
y_test_temp = scaler_temp_y.transform(y_test_temp)

# Prepare Regressor for multiple tasks learning
regressor = tf.estimator.Estimator(model_fn=lstm_model, model_dir=r'C:\Users\ZHA244\Coding\tensorflow\logs',
                                   params=model_params)


x_train = x_train.reshape((x_train.shape[0], model_params['TIMESTEPS'], model_params['N_FEATURES']))

print(x_train.shape)
print(x_train)

x_test = x_test.reshape((x_test.shape[0],model_params['TIMESTEPS'],model_params['N_FEATURES']))
# Train.
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': x_train}, y=y_train_ntu, batch_size=model_params['BATCH_SIZE'], num_epochs=None, shuffle=False)
regressor.train(input_fn=train_input_fn, steps=model_params['TRAINING_STEPS'])

# Predict.


test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': x_test}, y=y_test_ntu, num_epochs=1,batch_size=model_params['BATCH_SIZE'], shuffle=False)
predictions = regressor.predict(input_fn=test_input_fn)

y_predicted = np.array(list(scaler_ntu_y.inverse_transform(p) for p in predictions))

# predicted = np.asmatrix(list(predictions),dtype = np.float64) #,as_iterable=False))



# Score with sklearn.
score_sklearn = mean_squared_error(scaler_ntu_y.inverse_transform(y_test_ntu), y_predicted)
print('RMSE (sklearn): {0:f}'.format(sqrt(score_sklearn)))
print("--------")
mae = mean_absolute_error(scaler_ntu_y.inverse_transform(y_test_ntu), y_predicted)
print("MAE (sklearn):{0:f}".format(mae))
print("---------")
r2 = r2_score(scaler_ntu_y.inverse_transform(y_test_ntu), y_predicted)
print("R2 (sklearn):{0:f}".format(r2))

# Score with tensorflow
rmse_tf = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(scaler_ntu_y.inverse_transform(y_test_ntu), y_predicted))))
with tf.Session() as sess:
    print('RMSE (tensorflow):{0:f}'.format(sess.run(rmse_tf)))


# Drawing

axis_data = getdate_index(filepath,2928,24)

ax = plt.gca()
xfmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
ax.xaxis.set_major_formatter(xfmt)

true_line, = plt.plot_date(axis_data, scaler_ntu_y.inverse_transform(y_test_ntu)[0:24], 'b-', color='blue',
                             label='True Value')
predict_line, = plt.plot_date(axis_data, np.array(y_predicted)[0:24], 'b-', color='Red',
                                label='Prediction Value')

plt.legend(handles=[true_line, predict_line])
plt.gcf().autofmt_xdate()
plt.show()


