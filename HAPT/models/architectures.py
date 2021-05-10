import gin
import tensorflow as tf
import logging

from .layers import unidirectional_LSTM_block, bidirectional_LSTM_block, cascaded_LSTM_block, gru_block, simpleRNN_block

@gin.configurable
def Architecture(n_classes, rnn_block_type, num_rnn_layers, rnn_units, dense_units, dropout_rate, window_size, features):

    blocks_dict = {
        'unidirectional': unidirectional_LSTM_block,
        'bidirectional': bidirectional_LSTM_block,
        'cascaded': cascaded_LSTM_block,
        'gru': gru_block,
        'simpleRNN': simpleRNN_block,
    }

    assert rnn_block_type in blocks_dict.keys(), f'Unkown lstm block type: {rnn_block_type}'

    num_rnn_layers = int(num_rnn_layers)
    rnn_units = int(rnn_units)
    dense_units = int(dense_units)

    RNN_block = blocks_dict[rnn_block_type]

    inputs = tf.keras.Input(shape=(window_size, features))
    out = RNN_block(inputs, rnn_units, num_rnn_layers)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='Model')

