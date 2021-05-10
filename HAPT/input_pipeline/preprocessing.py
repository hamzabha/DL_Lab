import gin
import tensorflow as tf
import logging


def normalization(df):
    return (df - df.mean(axis=0)) / df.std(axis=0)


@tf.autograph.experimental.do_not_convert
@gin.configurable
def windowing(dataset, batch_size, window_size, shift, stride, buffer_size,
              repeat=False, shuffle=False, for_visualization=False, remove_unlabelled=False):
    if for_visualization:
        shift = window_size

    ds = tf.data.Dataset.from_tensor_slices(dataset)
    window = ds.window(window_size, shift=shift, stride=stride, drop_remainder=True)
    window = window.flat_map(lambda w: w.batch(window_size)) # maps the set of datasets into series of tensors
    
    if remove_unlabelled and not for_visualization:
        # filter out windows with unlabelled data
        window = window.filter(lambda w: tf.reduce_min(w[:, -1]) >= 0) 

    # mapping the window to a sequence of features and a one hot encoded label
    window = window.map(lambda w: (w[:, :-1], tf.one_hot(tf.cast(w[:, -1], 'int64'), depth=12))) 
    if shuffle and not for_visualization:
        window = window.shuffle(buffer_size)

    if for_visualization:
        # to have one batch contian approx. one entire experience
        window = window.batch(80)
        # to get a random experience each time visualization.py is run
        window = window.shuffle(buffer_size)
    else:
        window = window.batch(batch_size)

    if repeat and not for_visualization:
        window = window.repeat(-1)
    window = window.prefetch(tf.data.experimental.AUTOTUNE)

    return window
