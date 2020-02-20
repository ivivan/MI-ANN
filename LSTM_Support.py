import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn as tflearn
from tensorflow.contrib import layers as tflayers
from tensorflow.contrib import rnn
import warnings

warnings.filterwarnings("ignore")


def lstm_model(features, labels, mode, params): # [Ftrl, Adam, Adagrad, Momentum, SGD, RMSProp]
    """
        Creates a deep model based on:
            * stacked lstm cells
            * an optional dense layers
        :param num_units: the size of the cells.
        :param rnn_layers: list of int or dict
                             * list of int: the steps used to instantiate the `BasicLSTMCell` cell
                             * list of dict: [{steps: int, keep_prob: int}, ...]
        :param dense_layers: list of nodes for each layer
        :return: the model definition
        """

    def lstm_cells(layers):
        if isinstance(layers[0], dict):
            return [rnn.DropoutWrapper(rnn.LSTMCell(layer['num_units'],state_is_tuple=True,initializer=tf.orthogonal_initializer()),layer['keep_prob'])
                    if layer.get('keep_prob')
                    else rnn.LSTMCell(layer['num_units'], state_is_tuple=True,initializer=tf.orthogonal_initializer())
                    for layer in layers]

        return [rnn.LSTMCell(steps, state_is_tuple=True,initializer=tf.orthogonal_initializer()) for steps in layers]

    def dnn_layers(input_layers, layers):
        if layers and isinstance(layers, dict):
            return tflayers.stack(input_layers, tflayers.fully_connected,
                                  layers['layers'],
                                  activation=layers.get('activation'),
                                  dropout=layers.get('dropout'))
        elif layers:
            return tflayers.stack(input_layers, tflayers.fully_connected, layers)
        else:
            return input_layers


    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells(params['RNN_LAYERS']), state_is_tuple=True)
    x_ = tf.unstack(features['x'], num=params['TIMESTEPS'], axis=1)
    output, layers = rnn.static_rnn(stacked_lstm, x_, dtype=dtypes.float64)
    output = dnn_layers(output[-1], params['DENSE_LAYERS'])

    # 通过无激活函数的全连接层计算线性回归，并将数据压缩成一维数组结构
    predictions = tf.contrib.layers.fully_connected(output, 1, None)

    # For multi task learning purpose, here we should define multiple full connected output layers?

    



    # 将predictions和labels调整统一的shape
    # labels = tf.reshape(labels, [-1])
    # predictions = tf.reshape(predictions, [-1])

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)

    loss = tf.losses.mean_squared_error(predictions, labels)

    train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(),
                                             optimizer="Adagrad",
                                             learning_rate=tf.train.exponential_decay(0.01, tf.contrib.framework.get_global_step(), decay_steps = 1000, decay_rate = 0.9, staircase=False, name=None))
                                             # learning_rate=0.01)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels,predictions)
    }

    # return EstimatorSpec
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,predictions=predictions,train_op=train_op,eval_metric_ops=eval_metric_ops)





