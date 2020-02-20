import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from math import sqrt
from data_process import data_io_do as dataio
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


tf.logging.set_verbosity(tf.logging.INFO)
# np.random.seed(1337)
# Prepare data



def data_together(filepath):

    csvs = []
    dfs = []

    for subdir, dirs, files in os.walk(filepath):
        for file in files:
            # print os.path.join(subdir, file)
            filepath = subdir + os.sep + file
            if filepath.endswith(".hdf5"):
                csvs.append(filepath)

    # if len(csvs) < 0:
    #     print('Need all QLD Gov sensor data files')
    # else:
    #     for f in csvs:
    #         dfs.append(pd.read_csv(f))

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



def rsquare(y_true,y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


# Parameters
model_params = {
    'TIMESTEPS': 12,
    'N_FEATURES':5,
    'train_no': 1839,
    'test_no':355
    # 'RNN_LAYERS': [{'num_units': 400}],
    # # 'RNN_LAYERS': [{'num_units': 60, 'keep_prob': 0.75},{'num_units': 120, 'keep_prob': 0.75},{'num_units': 60, 'keep_prob': 0.75}],
    # 'DENSE_LAYERS': None,
    # 'TRAINING_STEPS': 15000,
    # 'PRINT_STEPS': 50,
    # 'BATCH_SIZE': 100
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

# filepath = r'C:\Users\ZHA244\Coding\QLD\baffle_creek\baffle-creek-buoy-quality-2013-all-forpca-120min.csv'
# x, y = generate_data(filepath, 732,model_params['TIMESTEPS'], 0, 'train', scaler_dic)
filepath = r'C:\Users\ZHA244\Coding\QLD\baffle_creek\dry_season-90min.csv'
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

# model = Sequential()
# model.add(Dense(3, input_dim=model_params['TIMESTEPS']* model_params['N_FEATURES'], activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(3, activation='relu'))
# # model.add(Dropout(0.2))
# model.add(Dense(3, activation='relu'))
# model.add(Dense(1, activation=None))
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='mse', optimizer='sgd',metrics=['mse','mae',rsquare])


##
# Training

# model_file = os.path.join('./models', '-{val_rsquare:.4f}-{epoch:02d}' + '.hdf5')
# checkpoint = ModelCheckpoint(filepath=model_file, monitor='val_rsquare', save_best_only=False,
#                                      save_weights_only=False, mode='max', period=1)
#
# history = model.fit(x_train, y_train_do,
#           epochs=1000,
#           batch_size=10,shuffle=True,validation_split=0.1,callbacks=[checkpoint])
# # # #

# trained_model = []
#
# trained_model = data_together(r'C:\Users\ZHA244\Coding\experiment\forfirstconf\train_dry120')
#
# for i in trained_model:
#     model = load_model(i, custom_objects={'rsquare': rsquare})
#     # score = model.evaluate(x_test, y_test_do, batch_size=1)
#     predictions = model.predict(x_test)
#
#     y_predicted = np.array(list(scaler_do_y.inverse_transform(p) for p in predictions))
#
#     # y_predicted = np.array(list(scaler_do_y.inverse_transform(p['predictions']) for p in predictions))
#     y_predicted = y_predicted.reshape(np.array(y_test_do).shape)
#
#     print("---------")
#     r2 = r2_score(scaler_do_y.inverse_transform(y_test_do), y_predicted)
#     print("File Name{0}-R2 (sklearn):{1:f}".format(i,r2))




model = load_model('save/dry-90.hdf5',custom_objects={'rsquare': rsquare})
score = model.evaluate(x_test, y_test_do, batch_size=1)

print(score)

predictions = model.predict(x_test)


y_predicted = np.array(list(scaler_do_y.inverse_transform(p) for p in predictions))

# y_predicted = np.array(list(scaler_do_y.inverse_transform(p['predictions']) for p in predictions))
y_predicted = y_predicted.reshape(np.array(y_test_do).shape)



# Score with sklearn.
score_sklearn = mean_squared_error(y_predicted, scaler_do_y.inverse_transform(y_test_do))
print('RMSE (sklearn): {0:f}'.format(sqrt(score_sklearn)))

print("--------")
mae = mean_absolute_error(scaler_do_y.inverse_transform(y_test_do), y_predicted)
print("MAE (sklearn):{0:f}".format(mae))
print("---------")
r2 = r2_score(scaler_do_y.inverse_transform(y_test_do), y_predicted)
print("R2 (sklearn):{0:f}".format(r2))






# Drawing

axis_data = getdate_index(filepath,model_params['train_no'],model_params['test_no'])
# # axis_data = getdate_index(filepath,976,496)
#
# ax = plt.gca()
# xfmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
# ax.xaxis.set_major_formatter(xfmt)

# true_line, = plt.plot_date(axis_data, scaler_do_y.inverse_transform(y_test_do)[0:336], 'b-', color='blue',
#                              label='True Value')
# predict_line, = plt.plot_date(axis_data, np.array(y_predicted)[0:336], 'b-', color='Red',
#                                 label='Prediction Value')
#
# plt.legend(handles=[true_line, predict_line])
# plt.gcf().autofmt_xdate()
# plt.show()

# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

# #
# # # for 90min
# # print('create new df')
# #
# # print(type(scaler_do_y.inverse_transform(y_test_do)[0:model_params['test_no']]))
# #
# result_df = pd.DataFrame(scaler_do_y.inverse_transform(y_test_do)[0:model_params['test_no']],columns=['Measured Values'])
# result_df['Prediction Values'] = np.array(y_predicted)[0:model_params['test_no']]
# print(result_df)
# print('end work')
#
#
ax = plt.gca()
# sns.set(font_scale=1.4)
# # sns.set(rc={'axes.facecolor':'white'})
# sns.lmplot(x='Measured Values', y='Prediction Values', data=result_df, ci=None, palette="muted", size=6,
#            scatter_kws={"s": 10, "alpha": 1})
# # plt.text(0,650, "R2 = 0.86085", fontsize = 20, color='black', fontstyle='italic')
# plt.title("R2 = 0.80498")
# # # plt.savefig(r'C:\Users\ZHA244\Pictures\paper-figure\120new.png', dpi=800,facecolor='white')
# # #
# plt.show()


#
#
xfmt = mdates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(xfmt)


true_line, = plt.plot_date(axis_data, scaler_do_y.inverse_transform(y_test_do), '-', lw=1, color=tableau20[2],
                         label='True Value')
predict_line, = plt.plot_date(axis_data, np.array(y_predicted), '--', lw=1, color=tableau20[18],
                           label='Prediction Value')


plt.legend(handles=[true_line, predict_line], fontsize=12)
plt.title('Water Quality Prediction', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('DO (mg/l)', fontsize=14)
plt.gcf().autofmt_xdate()
# plt.savefig(r'C:\Users\ZHA244\Pictures\paper-figure\90min-7days.png', dpi=200)
plt.show()

