import numpy as np
import pandas as pd
from sklearn import preprocessing


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


def load_csvdata(rawdata, time_steps,mode,scalardic):
    data = rawdata
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    train_x = rnn_data(data['Turbidity_NTU'], time_steps, labels=False)
    train_y = rnn_data(data['Turbidity_NTU'], time_steps, labels=True)

    train_x_two = rnn_data(data['EC_uScm'], time_steps, labels=False)
    train_y_two = rnn_data(data['EC_uScm'], time_steps, labels=True)

    train_x_three = rnn_data(data['Temp_degC'], time_steps, labels=False)
    train_y_three = rnn_data(data['Temp_degC'], time_steps, labels=True)

    train_x = np.squeeze(train_x)
    train_x_two = np.squeeze(train_x_two)
    train_x_three = np.squeeze(train_x_three)

    # Scale data (training set) to 0 mean and unit standard deviation.
    if(mode=='train'):
        scaler_ntu = scalardic['scaler_one']
        scaler_ec = scalardic['scaler_two']
        scaler_temp = scalardic['scaler_three']

        train_x = scaler_ntu.fit_transform(train_x)
        train_x_two = scaler_ec.fit_transform(train_x_two)
        train_x_three = scaler_temp.fit_transform(train_x_three)
    elif (mode=='test'):
        scaler_ntu = scalardic['scaler_one']
        scaler_ec = scalardic['scaler_two']
        scaler_temp = scalardic['scaler_three']

        train_x = scaler_ntu.transform(train_x)
        train_x_two = scaler_ec.transform(train_x_two)
        train_x_three = scaler_temp.transform(train_x_three)

    all_train = np.stack((train_x, train_x_two, train_x_three), axis=-1)
    all_train = all_train.reshape(-1,time_steps*3)

    return dict(train=all_train,scalerone=scaler_ntu,scalertwo=scaler_ec,scalerthree=scaler_temp), dict(trainyone=train_y,trainytwo=train_y_two,trainythree=train_y_three)


#data input
# filepath = r'C:\Users\ZHA244\Coding\QLD\baffle_creek\baffle-creek-buoy-quality-2013-multiple.csv'
#
# dataset = pd.read_csv(filepath) #384 5 days
# dataset['TIMESTAMP'] = pd.to_datetime(dataset['TIMESTAMP'],dayfirst=True)
# dataset['TimeNumber'] = dataset['TIMESTAMP'].apply(lambda x: x.timestamp()).values
# timesteps = 2
#
#
# sets_x, sets_y = load_csvdata(dataset,timesteps)
#
#
# for k,v in sets_x.items():
#     print(k)
#     print(v)