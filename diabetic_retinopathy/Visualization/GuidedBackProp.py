import tensorflow as tf
import numpy as np

@tf.custom_gradient
def guided_relu(x):
    y = tf.nn.relu(x)

    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy

    return y, grad


class GuidedBackProp:

    def __init__(self, model):

        self.model = model
        self.layer_name = self.findlayer()
        self.new_model = self.modify_model()

    def findlayer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name

    def modify_model(self):
        """ Function that changes the gradient function for all ReLU activations """

        new_model = tf.keras.models.Model(
            inputs=[self.model.input],
            outputs=[self.model.get_layer(self.layer_name).output]
        )

        # find the layers with activation functions
        layers = [layer for layer in new_model.layers[1:]
                  if hasattr(layer, 'activation')]

        # overwrite the relu layers
        for layer in layers:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guided_relu

        return new_model

    def guided_backprop(self, image: np.ndarray):

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            tape.watch(inputs)
            outputs = self.new_model(inputs)

        grads = tape.gradient(outputs, inputs)[0]
        return grads