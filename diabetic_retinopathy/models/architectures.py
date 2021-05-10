import gin
import tensorflow as tf
import tensorflow_hub as hub
import logging

from .layers import CNN_block, InceptionResnetV2

@gin.configurable
def Architecture(input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate, model_type, fine_tuning=False):
    if type(model_type) is str:
        return tf_hub_architecture(model_type, input_shape, n_classes, dropout_rate, fine_tuning)

    # allow floats as inputs (for bayesian optimization by tune), and turn them into integers here
    base_filters = int(base_filters)
    n_blocks = int(n_blocks)
    dense_units = int(dense_units)

    assert n_blocks > 0, 'Number of blocks has to be at least 1.'

    inputs = tf.keras.Input(input_shape)

    if model_type == CNN_block:
        out = model_type(inputs, base_filters)
        out = tf.keras.layers.MaxPool2D((2, 2))(out)
        for i in range(2, n_blocks-1):
            out = model_type(out, base_filters * 2 ** (i))
            out = tf.keras.layers.MaxPool2D((2, 2))(out)

        out = model_type(out, base_filters * 2 ** n_blocks)

    elif model_type == InceptionResnetV2:
        out = model_type(inputs)

    else:
        out = model_type(inputs, base_filters)

    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='Model')

# recommended input sizes for some tf-hub models
expected_input_sizes = {
    'https://tfhub.dev/google/imagenet/inception_v1/feature_vector/4': [224, 224],
    'https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1': [380, 380],
    'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4': [224, 224],
}

def tf_hub_architecture(tf_hub_url, input_shape, n_classes, dropout_rate, fine_tuning=False):
    if not tf_hub_url in expected_input_sizes.keys():
        logging.warning('Unknown tf hub model')
    if expected_input_sizes[tf_hub_url] != input_shape[:2]:
        logging.warning(f"Input size {input_shape[:2]} doesn't match expected size of this tf-hub model: {expected_input_sizes[tf_hub_url]}")

    model = tf.keras.Sequential([
        hub.KerasLayer(tf_hub_url, trainable=fine_tuning),
        tf.keras.layers.Dropout(rate=dropout_rate),
        tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax),
    ])
    model.build([None] + input_shape)

    return model

