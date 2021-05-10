import matplotlib.pyplot as plt
import tensorflow as tf
from models.architectures import Architecture
from models.layers import CNN_block


def read_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.
    image = tf.image.crop_to_bounding_box(image, *(0, 275, 2848, 3415))
    image = tf.image.resize_with_pad(image, target_height=256, target_width=256)
    image = tf.expand_dims(image, axis=0)
    return image


# black baseline
baseline = tf.zeros(shape=(256, 256, 3))

def interpolation(baseline, image, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    delta = image - baseline_x
    images = baseline_x + alphas_x * delta
    return images


# compute the grads between model outputs and interpolated images
def compute_grads(images, target_class_idx, model):
    with tf.GradientTape() as tape:
        tape.watch(images)
        preds = model(images)
        probs = preds[:, target_class_idx]
    return tape.gradient(probs, images)


def integral_approximation(gradients):
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


@tf.autograph.experimental.do_not_convert
def integrated_gradients(baseline, image, model, target_class_idx, steps, batch_size=32):

    alphas = tf.linspace(start=0.0, stop=1.0, num=steps+1)

    # initialize tensorarray
    gradient_batches = tf.TensorArray(tf.float32, size=steps + 1)

    for alpha in range(0, alphas.shape[0], batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, alphas.shape[0])
        alpha_batch = alphas[from_:to]

        interpolated_path_input_batch = interpolation(baseline=baseline, image=image, alphas=alpha_batch)

        gradient_batch = compute_grads(images=interpolated_path_input_batch, model=model, target_class_idx=target_class_idx)

        gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)

    total_gradients = gradient_batches.stack()

    avg_gradients = integral_approximation(gradients=total_gradients)

    integrated_gradients = (image - baseline) * avg_gradients

    return integrated_gradients


def plot_image(baseline, image, model, target_class_idx, steps, cmap=None, attenuation=0.4):
    attributions = integrated_gradients(baseline, image, model, target_class_idx, steps)

    attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

    image = tf.squeeze(image)
    attribution_mask = tf.squeeze(attribution_mask)

    fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(12, 12))

    axs[0, 0].set_title('Baseline')
    axs[0, 0].imshow(baseline)
    axs[0, 0].axis('off')

    axs[0, 1].set_title('image')
    axs[0, 1].imshow(image)
    axs[0, 1].axis('off')

    axs[1, 0].set_title('Attribution mask')
    axs[1, 0].imshow(attribution_mask, cmap=cmap, )
    axs[1, 0].axis('off')

    axs[1, 1].set_title('overlayed image')
    axs[1, 1].imshow(attribution_mask, cmap=cmap)
    axs[1, 1].imshow(image, alpha=attenuation)
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()
    return fig
