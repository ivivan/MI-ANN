import numpy as np
import pandas as pd


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


def split_data(data, val_size=0, test_size=0):
    """
    splits data to training, validation and testing parts
    """
    #split based on percentage
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))

    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]

    return df_train, df_val, df_test


def prepare_data(data, time_steps, labels=False, val_size=0, test_size=0):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    """
    df_train, df_val, df_test = split_data(data, val_size, test_size)

    return (rnn_data(df_train, time_steps, labels=labels),
            rnn_data(df_val, time_steps, labels=labels),
            rnn_data(df_test, time_steps, labels=labels))


def load_csvdata(rawdata, time_steps, seperate=False):
    data = rawdata
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)


    train_x, val_x, test_x = prepare_data(data['Turbidity_NTU'] if seperate else data, time_steps)
    train_y, val_y, test_y = prepare_data(data['Turbidity_NTU'] if seperate else data, time_steps, labels=True)

    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)


def generate_data(fct, x, time_steps, seperate=False):
    """generates data with based on a function fct"""
    data = fct(x)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    train_x, val_x, test_x = prepare_data(data['Turbidity_NTU'] if seperate else data, time_steps)
    train_y, val_y, test_y = prepare_data(data['Turbidity_NTU'] if seperate else data, time_steps, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)
















#
# def parser(x):
#     return pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
#
#
# series = pd.read_csv(r'C:\Users\ZHA244\Coding\QLD\baffle_creek\baffle-creek-buoy-quality-2013-turbidity-oneweek.csv', sep=',', usecols=[1,2],
#                   parse_dates=[0], header=0, index_col=0, squeeze=True, date_parser=parser)


# # data input
# filepath = r'C:\Users\ZHA244\Coding\QLD\baffle_creek\baffle-creek-buoy-quality-2013-turbidity-training_test_together.csv'
#
# dataset = pd.read_csv(filepath) #384 5 days
# dataset['TIMESTAMP'] = pd.to_datetime(dataset['TIMESTAMP'],dayfirst=True)
# dataset['TimeNumber'] = dataset['TIMESTAMP'].apply(lambda x: x.timestamp()).values
# timesteps = 144
#
# #print(dataset)
#
# sets_x, sets_y = load_csvdata(dataset,timesteps,seperate=True)
#
# for k,v in sets_x.items():
#     print(k)
#     print(len(v))