import gin
import tensorflow as tf
import numpy as np

@gin.configurable
def preprocess(image, label, img_size, relevant_bbox=None):
    """Dataset preprocessing: Normalizing and resizing"""

    # Normalize image: `uint8` -> `float32`.
    image = tf.cast(image, tf.float32) / 255.

    if relevant_bbox:
        image = tf.image.crop_to_bounding_box(image, *relevant_bbox)
    image = tf.image.resize_with_pad(image, *img_size)

    return image, label

@gin.configurable
def augment(image, label, min_cropping_size, max_cropping_size, cropping_probability, flip, rotate, brightness, brightness_max_delta,
            contrast, contrast_min_factor, hue, hue_max_delta, saturation, saturation_min_factor):
    """Data augmentation"""

    if tf.random.uniform(tuple()) < cropping_probability:
        origin_shape = tf.shape(image)

        tau = tf.random.uniform([])
        cropping_size = (
            int(min_cropping_size + (max_cropping_size - min_cropping_size)*tau),
            int(min_cropping_size + (max_cropping_size - min_cropping_size)*tau)
        )
        
        image = tf.image.random_crop(image, (*cropping_size, 3))
        image = tf.image.resize(image, origin_shape[:2])

    if flip:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

    if rotate:
        image = tf.image.rot90(image, tf.random.uniform(tuple(), maxval=4, dtype=tf.int32))

    if brightness:
        image = tf.image.random_brightness(image, brightness_max_delta)

    if contrast:
        image = tf.image.random_contrast(image, contrast_min_factor, 1/contrast_min_factor)

    if hue:
        image = tf.image.random_hue(image, hue_max_delta)

    if saturation:
        image = tf.image.random_saturation(image, saturation_min_factor, 1/saturation_min_factor)


    return image, label