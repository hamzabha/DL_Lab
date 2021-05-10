import tensorflow as tf
import numpy as np
import os
import logging
from random import sample

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def image_example(image_string, label):
    feature = {
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

# write the data of the list of tuples image_labels as a tfrecord file to the specified path
def write_tfrecord(image_labels, path):
    with tf.io.TFRecordWriter(path) as writer:
        for filename, label in image_labels:
            image_string = open(filename, 'rb').read()
            tf_example = image_example(image_string, label)
            writer.write(tf_example.SerializeToString())


def create_tfrecords(dataset_source_dir, tfrecord_target_dir, val_split, label_mapping=None, resampling=True):
    """Creates TFRecord files from a dataset of raw images and labels and saves them at `tfrecord_target_dir`.
       Simply returns without an Exception, if the tfrecords for this dataset already exist.

    Parameters:
        dataset_source_dir (string): Directory of the raw dataset. The only currently accepted structure of this directory is the following:
        ├── images/
        │   ├── test/
        │   │   ├── <test_image_1_filename>.jpg
        │   │   ├── ...
        │   │   └── <test_image_N_filename>.jpg
        │   └── train/
        │       ├── <train_image_1_filename>.jpg
        │       ├── ...
        │       └── <train_image_M_filename>.jpg
        └── labels/
            ├── test.csv    # First column: <test_image_n_filename> (string),  Second column: label (int) of the given image
            └── train.csv   # First column: <train_image_m_filename> (string), Second column: label (int) of the given image

        tfrecord_target_dir (string): Target directory of the TFRecord-dataset created by this function. Will be created if it does not exist.
        val_split (string): validation split: how much of the training data will be used as validation data
        label_mapping (function: int -> int), optional: a function to map labels onto new labels
        resampling (bool): Whether to balance the dataset using resampling. Only has an effect if the dataset is unbalanced.
    """
    
    # return if the tfrecords for this dataset have already been created
    if (os.path.isfile(os.path.join(tfrecord_target_dir, 'test.tfrecords')) or
        os.path.isfile(os.path.join(tfrecord_target_dir, 'train.tfrecords')) or
        os.path.isfile(os.path.join(tfrecord_target_dir, 'val.tfrecords'))):
        print(logging.info(f'Overwriting existing TFRecord-files at {tfrecord_target_dir}'))

    # Load image file paths and corresponding labels from label-csv-files
    with open(os.path.join(dataset_source_dir, 'labels', 'train.csv')) as train_labels_file:
        train_filenames_labels = [
            (
                os.path.join(dataset_source_dir, 'images', 'train', line.split(',')[0] + '.jpg'), # image path
                label_mapping(int(line.split(',')[1])) if label_mapping else int(line.split(',')[1]) # image label
            )
            for line in train_labels_file.readlines()[1:] # skip header line
        ]

    # Load filenames and labels of test data
    with open(os.path.join(dataset_source_dir, 'labels', 'test.csv')) as test_labels_file:
        test_filenames_labels = [
            (
                os.path.join(dataset_source_dir, 'images', 'test', line.split(',')[0] + '.jpg'), # image path
                label_mapping(int(line.split(',')[1])) if label_mapping else int(line.split(',')[1]) # image label
            )
            for line in test_labels_file.readlines()[1:] # skip header line
        ]

    # Separate training data into train and validation data
    split_index = round(len(train_filenames_labels) * (1-val_split))
    val_filenames_labels = train_filenames_labels[split_index:]
    train_filenames_labels = train_filenames_labels[:split_index]

    if resampling:
        # Determine unique labels
        train_labels = [label for filename, label in train_filenames_labels]
        unique_labels = sorted(list(set(train_labels)))

        # Helper function for resampling
        def get_class_samples(label, num_samples):
            samples = [(f, l) for f, l in train_filenames_labels if l == label]
            return samples * (num_samples//len(samples)) + sample(samples, num_samples % len(samples))

        # Resample to balance class occurences
        class_cardinalities = [train_labels.count(label) for label in unique_labels]
        train_filenames_labels += [
            sample
            for idx, label in enumerate(unique_labels)
            for sample in get_class_samples(label, max(class_cardinalities)-class_cardinalities[idx]) 
        ]

    os.makedirs(tfrecord_target_dir, exist_ok=True)
    write_tfrecord(train_filenames_labels, os.path.join(tfrecord_target_dir, 'train.tfrecords'))
    write_tfrecord(val_filenames_labels, os.path.join(tfrecord_target_dir, 'val.tfrecords'))
    write_tfrecord(test_filenames_labels, os.path.join(tfrecord_target_dir, 'test.tfrecords'))


def load_tfrecord_dataset(tfrecord_path):
    raw_image_dataset = tf.data.TFRecordDataset(tfrecord_path)

    # Create a dictionary describing the features.
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_image_function(example_proto):
        # Parse the input tf.train.Example proto using the dictionary above.
        parsed_example = tf.io.parse_single_example(example_proto, image_feature_description)
        image_string = parsed_example['image_raw']
        return tf.io.decode_jpeg(image_string), parsed_example['label']

    return raw_image_dataset.map(_parse_image_function)
