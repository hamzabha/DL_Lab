import gin
import logging
import os
from random import sample
import tensorflow as tf
import tensorflow_datasets as tfds

from input_pipeline.tfrecords_handler import create_tfrecords, load_tfrecord_dataset
from input_pipeline.preprocessing import preprocess, augment

@gin.configurable
def load(name, data_dir, val_split_idrid):
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")

        # {0, 1} -> 0, {2, 3, 4} -> 1
        def label_mapping(label):
            return int(label > 1)
        num_classes = 2

        tfrecord_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'tfrecord_datasets', name)

        if not os.path.isdir(tfrecord_dir):
            create_tfrecords(os.path.join(data_dir, 'IDRID_dataset'), tfrecord_dir, val_split_idrid, label_mapping)

        ds_train = load_tfrecord_dataset(os.path.join(tfrecord_dir, 'train.tfrecords'))
        ds_val = load_tfrecord_dataset(os.path.join(tfrecord_dir, 'val.tfrecords'))
        ds_test = load_tfrecord_dataset(os.path.join(tfrecord_dir, 'test.tfrecords'))

        ds_info = {
            'name': name,
            'num_training_samples':  sum(1 for _ in ds_train),
            'num_test_samples': sum(1 for _ in ds_test),
            'num_classes': num_classes,
            'features_shape': [2848, 4288, 3],
        }

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'diabetic_retinopathy_detection/btgraham-300',
            split=['train', 'validation', 'test'],
            shuffle_files=True,
            with_info=True,
            data_dir=os.path.join(data_dir, 'tensorflow_datasets')
        )

        def _preprocess(img_label_dict):
            return img_label_dict['image'], img_label_dict['label']

        ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds_info = {
            'name': name,
            'num_training_samples': ds_info.splits['train'].num_examples,
            'num_test_samples': ds_info.splits['test'].num_examples,
            'num_classes': ds_info.features["label"].num_classes,
            'features_shape': list(ds_info.features["image"].shape),
        }

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "mnist":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train[:90%]', 'train[90%:]', 'test'],
            shuffle_files=True,
            with_info=True,
            as_supervised=True,
            data_dir=os.path.join(data_dir, 'tensorflow_datasets')
        )

        ds_info = {
            'name': name,
            'num_training_samples': ds_info.splits['train'].num_examples,
            'num_test_samples': ds_info.splits['test'].num_examples,
            'num_classes': ds_info.features["label"].num_classes,
            'features_shape': list(ds_info.features["image"].shape),
        }

        return prepare(ds_train, ds_val, ds_test, ds_info)

    else:
        raise ValueError

@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching, img_size):
    # Update the shape to the dimensions the images are being resized to
    ds_info['features_shape'][:2] = img_size
    # Prepare training dataset
    ds_train = ds_train.map(
        lambda image, label: preprocess(image, label, img_size),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    if caching:
        assert caching in ('disk', 'memory'), 'Parameter `caching` must be one of (False, "disk", "memory").'
        if caching == 'disk':
            os.makedirs(
                os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'dataset_cache', ds_info['name']), exist_ok=True
            )
        ds_train = ds_train.cache(
            '' if caching == 'memory'
            else os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'dataset_cache', ds_info['name'], 'train_cache')
        )
    ds_train = ds_train.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(ds_info['num_training_samples'] // 10)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(
        lambda image, label: preprocess(image, label, img_size),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache(
            '' if caching == 'memory'
            else os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'dataset_cache', ds_info['name'], 'val_cache')
        )
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        lambda image, label: preprocess(image, label, img_size),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache(
            '' if caching == 'memory'
            else os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'dataset_cache', ds_info['name'], 'test_cache')
        )
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info