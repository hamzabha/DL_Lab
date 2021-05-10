import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from absl import app, flags
import os
import logging
import gin
from random import sample

from Visualization.GradCam import GradCam
from Visualization.GuidedBackProp import GuidedBackProp
from Visualization.IntegratedGrad import integrated_gradients
from models.architectures import Architecture
from models.layers import CNN_block
from utils import utils_params, utils_misc

# to avoid errors with gin
from input_pipeline import preprocessing, datasets
from train import Trainer

FLAGS = flags.FLAGS
flags.DEFINE_string('mode', 'gradcam', 'Specify the visualization technique to be used.  One of ("gradcam", "integrated_gradients")')


flags.DEFINE_string('image_path', '',  
    "Specify the path of the image to be tested, or don't specify anything to use random images of the test data set."
)

flags.DEFINE_integer('num_classes', 2,
    ('Specify the number of classes the model is built to classify. Only required if --image_path is specified. '
     '(Otherwise it is implied by the dataset).')
)

flags.DEFINE_integer('num_images', 1,  
    ("Specify the number of images to be visualized. Only takes effect if --image_path is not specified, "
     "and images are randomly sampled from the test data set instead.")
)

flags.DEFINE_string('visualize_from', '',
    'Specify either the exact checkpoint inside a run folder, or the run folder itself to use its latest checkpoint for visualization.'
)


def load_image(image_path):
    image_string = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image_string)
    image = tf.cast(image, tf.float32) / 255.

    image = tf.image.crop_to_bounding_box(image, 0, 275, 2848, 3415)
    image = tf.image.resize_with_pad(image, 256, 256)

    image = tf.expand_dims(image, axis=0)
    return image


def deprocess_image(image):
    image = image.copy()
    image -= image.mean()
    image /= (image.std() + 1e-07)
    image *= 0.25
    image += 0.5
    image = np.clip(image, 0, 1)
    image = 255 * image
    image = np.clip(image, 0, 255).astype('uint8')

    return image

def main(argv):
    assert FLAGS.visualize_from, 'Need to specify a checkpoint or run folder to load the model'
    assert FLAGS.mode in ('gradcam', 'integrated_gradients'), 'Unkown mode'

    # if run folder is specified, use it directly, otherwise it's assumed that the checkpoint is specified, so extract the run folder (2 directories above)
    run_folder = FLAGS.visualize_from if os.path.isdir(FLAGS.visualize_from) else os.path.join(os.path.dirname(FLAGS.visualize_from), os.pardir)

    # reuse existing folder structures
    run_paths = utils_params.gen_run_folder(run_folder)
    # set loggers (appending to existing log file)
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
    # reuse the saved gin operative of the run folder as config
    gin.parse_config_files_and_bindings([run_paths['path_gin']], [])

    if FLAGS.image_path:
        image = load_image(FLAGS.image_path)
        model = load_model(image.shape[1:], FLAGS.num_classes, checkpoint_or_folder=FLAGS.visualize_from)

        predictions = model.predict(image)
        prediction = tf.argmax(predictions[0])
        print('\nPrediction:  ', prediction.numpy())
        if FLAGS.mode == 'gradcam':
            show_gradcam(model, image)
        else:
            show_integrated_gradients(model, image, prediction)
    else:
        # setup pipeline
        _, _, ds_test, ds_info = datasets.load()

        # un-batch ds into list of single images, labels
        images, labels = zip(*ds_test)
        unbatched_images = np.concatenate(images, axis=0)
        unbatched_labels = np.concatenate(labels, axis=0)
        dataset_list = list(zip(unbatched_images, unbatched_labels))

        model = load_model(input_shape=ds_info['features_shape'], n_classes=ds_info['num_classes'], checkpoint_or_folder=FLAGS.visualize_from)

        for image, label in sample(dataset_list, FLAGS.num_images):
            image = np.expand_dims(image, axis=0)
            predictions = model.predict(image)
            prediction = tf.argmax(predictions[0])
            print('\nPrediction:  ', prediction.numpy())
            print('Ground Truth:', label, end='\n\n')
            if FLAGS.mode == 'gradcam':
                show_gradcam(model, image)
            else:
                show_integrated_gradients(model, image, prediction)

def load_model(input_shape, n_classes, checkpoint_or_folder):
    model = Architecture(input_shape=input_shape, n_classes=n_classes)

    checkpoint = tf.train.Checkpoint(model=model)
    if os.path.isdir(checkpoint_or_folder):
        checkpoint.restore(
            tf.train.CheckpointManager(checkpoint, run_paths["path_ckpts_train"], max_to_keep=None).latest_checkpoint
        )
    else:
        checkpoint.restore(checkpoint_or_folder)

    return model

def show_gradcam(model, image):
    gradcam = GradCam(model)
    Guidedbp = GuidedBackProp(model)
    heatmap = gradcam.heat_map(image)
    cam = heatmap * 0.4 + image

    guided_backprop = Guidedbp.guided_backprop(image)
    guided_gradcam = guided_backprop * np.repeat(heatmap, 1, axis=2)

    image = tf.squeeze(image)
    cam = tf.squeeze(cam)

    fig, axs = plt.subplots(nrows=2, ncols=3, squeeze=False, figsize=(12, 12))

    axs[0, 0].set_title('image')
    axs[0, 0].imshow(image)
    axs[0, 0].axis('off')

    axs[0, 1].set_title('Heatmap')
    gcam = axs[0, 1].imshow(tf.squeeze(heatmap))
    axs[0, 1].axis('off')
    fig.colorbar(gcam, ax=axs[0, 1])

    axs[0, 2].set_title('GradCam')
    gcam = axs[0, 2].imshow(cam)
    axs[0, 2].axis('off')
    fig.colorbar(gcam, ax=axs[0, 2])

    axs[1, 0].set_title('Guided Backpropagation')
    axs[1, 0].imshow(np.flip(deprocess_image(np.array(guided_backprop)), -1))
    axs[1, 0].axis('off')

    axs[1, 1].set_title('Guided GradCam')
    axs[1, 1].imshow(np.flip(deprocess_image(np.array(guided_gradcam)), -1))
    axs[1, 1].axis('off')

    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

def show_integrated_gradients(model, image, prediction):
    baseline = tf.zeros(shape=image.shape[1:])

    attributions = integrated_gradients(baseline, image, model, prediction, steps=240)

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
    axs[1, 0].imshow(attribution_mask, cmap=plt.cm.get_cmap('jet'))
    axs[1, 0].axis('off')

    axs[1, 1].set_title('Integrated gradients')
    axs[1, 1].imshow(attribution_mask, cmap=plt.cm.get_cmap('jet'))
    axs[1, 1].imshow(image, alpha=0.4)
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    app.run(main)
