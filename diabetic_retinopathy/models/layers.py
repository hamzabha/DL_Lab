import gin
import tensorflow as tf

@gin.configurable
def CNN_block(inputs, filters, kernel_size):

    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(inputs)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)
    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(out)
    out = tf.keras.layers.BatchNormalization()(out)

    return out


def InceptionResnetV2(inputs):

    out = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top = False, weights = 'imagenet')(inputs)


    return out

@gin.configurable
def Hybrid(inputs,units):

    cnn = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top = False, weights = 'imagenet')(inputs)

    rnn = tf.keras.layers.LSTM(units)(inputs)
    rnn = tf.keras.layers.LSTM(2*units)(rnn)
    rnn = tf.keras.layers.LSTM(4*units)(rnn)

    out = tf.keras.layers.Merge(cnn, rnn, mode = 'concat')

    return out
