import gin
import tensorflow as tf


def unidirectional_LSTM_block(inputs, units, num_layers):
    '''A block of `num_layers` unidirectional LSTM layers with `units` units each.'''

    assert num_layers > 0, 'num_layers needs to be positive'
    out = inputs

    for _ in range(num_layers):
        out = tf.keras.layers.LSTM(units, return_sequences=True)(out)

    return out


def bidirectional_LSTM_block(inputs, units, num_layers):
    '''A block of `num_layers` bidirectional LSTM layers with `units` units each.'''

    assert num_layers > 0, 'num_layers needs to be positive'
    out = inputs

    for _ in range(num_layers):
        out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(out)

    return out


def cascaded_LSTM_block(inputs, units, num_layers):
    '''A block of one bidirectional, and then `num_layers` unidirectional LSTM layers with `units` units each.'''

    assert num_layers > 0, 'num_layers needs to be positive'

    out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(inputs)

    for _ in range(num_layers):
        out = tf.keras.layers.LSTM(units, return_sequences=True)(out)

    return out


def gru_block(inputs, units, num_layers):
    '''A block of `num_layers` GRU layers with `units` units each.'''

    assert num_layers > 0, 'num_layers needs to be positive'
    out = inputs

    for _ in range(num_layers):
        out = tf.keras.layers.GRU(units, return_sequences=True)(out)

    return out


def simpleRNN_block(inputs, units, num_layers):
    '''A block of `num_layers` SimpleRNN layers with `units` units each.'''

    assert num_layers > 0, 'num_layers needs to be positive'
    out = inputs

    for _ in range(num_layers):
        out = tf.keras.layers.SimpleRNN(units, return_sequences=True)(out)

    return out