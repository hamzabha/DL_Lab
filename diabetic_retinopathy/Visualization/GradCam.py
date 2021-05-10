import tensorflow as tf
import numpy as np
import matplotlib.cm as cm


class GradCam:

    def __init__(self, model):

        self.model = model
        self.last_conv_layer = self.findlayer()

    def findlayer(self):
        for layer in reversed(self.model.layers):
            if (len(layer.output_shape) == 4) and ('conv2d' in layer.name):
                return layer.name

    def heat_map(self, input):

        # Create our gradient model by supplying the final 4D layer and the output of the softmax activation from the original model
        new_model = tf.keras.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.last_conv_layer).output,
                     self.model.output])

        with tf.GradientTape() as tape:
            inputs = tf.cast(input, tf.float32)
            (last_conv_out, preds) = new_model(inputs, training=False)
            top_pred_idx = tf.argmax(preds[0])
            top_class = preds[:, top_pred_idx]

        # compute gradients with auto differentiation
        grads = tape.gradient(top_class, last_conv_out)

        # discard batch
        last_conv_out = last_conv_out.numpy()[0]
        grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        pooled_grads = grads.numpy()

        for i in range(pooled_grads.shape[-1]):
            last_conv_out[:, :, i] *= pooled_grads[i]
        # normalizing the heatmap between 0 and 1
        heatmap = np.mean(last_conv_out, axis=-1)
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap))

        # rescaling and colorizing the heatmap
        heatmap = 255 * heatmap
        heatmap = heatmap.astype('uint8')
        (w, h) = (input.shape[1], input.shape[2])
        dis = cm.get_cmap("jet")
        colour = dis(np.arange(256))[:, :3]
        dis_heat = colour[heatmap]
        dis_heat = tf.image.resize(dis_heat, [w, h])

        return dis_heat