import gradio as gr
import tensorflow as tf
import numpy as np
import os
import logging
import gin
from Visualization.GradCam import GradCam
from Visualization.GuidedBackProp import GuidedBackProp
from Visualization.IntegratedGrad import integrated_gradients
import matplotlib.pyplot as plt
from models.architectures import Architecture
from utils import utils_params, utils_misc
import sys

# to avoid errors with gin
from input_pipeline import datasets
from train import Trainer


checkpoint_path = os.path.join(sys.path[0], 'best_model/run_2020-12-25T09-53-02-345803', 'ckpts', 'ckpt-2')


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

def load_model(input_shape, n_classes, checkpoint_or_folder):
    model = Architecture(input_shape=input_shape, n_classes=n_classes)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(checkpoint_or_folder)

    return model

def load_image(inp):

    image = tf.cast(inp, tf.float32) / 255.

    image = tf.image.crop_to_bounding_box(image, 0, 275, 2848, 3415)
    image = tf.image.resize_with_pad(image, 256, 256)

    image = tf.expand_dims(image, axis=0)
    return image


# if run folder is specified, use it directly, otherwise it's assumed that the checkpoint is specified, so extract the run folder (2 directories above)
run_folder = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.join(os.path.dirname(checkpoint_path), os.pardir)


# reuse existing folder structures
run_paths = utils_params.gen_run_folder(run_folder)
# set loggers (appending to existing log file)
utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
# reuse the saved gin operative of the run folder as config
gin.parse_config_files_and_bindings([run_paths['path_gin']], [])

def show_visualization(model, image, prediction):
    gradcam = GradCam(model)
    Guidedbp = GuidedBackProp(model)
    heatmap = gradcam.heat_map(image)
    cam = heatmap * 0.4 + image
    guided_backprop = Guidedbp.guided_backprop(image)
    guided_gradcam = guided_backprop * np.repeat(heatmap, 1, axis=2)

    baseline = tf.zeros(shape=image.shape[1:])
    attributions = integrated_gradients(baseline, image, model, prediction, steps=240)
    attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

    image = tf.squeeze(image)
    cam = tf.squeeze(cam)
    heatmap = tf.squeeze(heatmap)
    attribution_mask = tf.squeeze(attribution_mask)

    fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False)
    axs[0, 0].set_title('preprocessed image')
    axs[0, 0].imshow(image)
    axs[0, 0].axis('off')

    axs[0, 1].set_title('GradCam')
    gcam = axs[0, 1].imshow(cam)
    axs[0, 1].axis('off')
    fig.colorbar(gcam, ax=axs[0, 1])

    axs[1, 0].set_title('Guided GradCam')
    axs[1, 0].imshow(np.flip(deprocess_image(np.array(guided_gradcam)), -1))
    axs[1, 0].axis('off')

    axs[1, 1].set_title('Integrated Gradients')
    axs[1, 1].imshow(attribution_mask, cmap=plt.cm.get_cmap('jet'))
    axs[1, 1].imshow(image, alpha=0.4)
    axs[1, 1].axis('off')

    if prediction == 1:
        label = 'Diabetic retinopathy has been detected'
    else :
        label = 'No diabetic retinopathy has been detected'

    plt.suptitle(label)

    return fig


def classify_image(input):
    image = load_image(input)
    model = load_model(image.shape[1:], 2, checkpoint_or_folder=checkpoint_path)
    predictions = model.predict(image)
    prediction = tf.argmax(predictions[0])

    plot = show_visualization(model, image, prediction)

    return plot


descriptive_text = "This UI offers the possibility to input a retina image and the model will predict whether or not the patient" \
                   " has diabetic retinopathy. Moreover, with the use of visualization techniques (Gradcam, " \
                   "guided-gradcam and integrated gradients), the suspected zone containing symptoms will be " \
                   "highlighted." \
                   "This model is EXPERIMENTAL and should only be used for research purposes.\n"\
                   "The Computation of the GradCam, Guided-GradCam and Integrated Gradients may take a few seconds."

image = gr.inputs.Image()
label = gr.outputs.Image(type="plot")

gr.Interface(fn=classify_image, inputs=image, outputs=label, title='Diabetic retinopathy detection',
             description=descriptive_text, capture_session=True, examples_per_page=5,
             examples=[[os.path.join(sys.path[0], "images/IDRiD_015.jpg")],
                       [os.path.join(sys.path[0], "images/IDRiD_091.jpg")],
                       [os.path.join(sys.path[0], "images/IDRiD_096.jpg")],
                       [os.path.join(sys.path[0], "images/IDRiD_053.jpg")],
                       [os.path.join(sys.path[0], "images/IDRiD_017.jpg")],
                       [os.path.join(sys.path[0], "images/IDRiD_033.jpg")],
                       [os.path.join(sys.path[0], "images/IDRiD_039.jpg")],
                       [os.path.join(sys.path[0], "images/IDRiD_049.jpg")],
                       [os.path.join(sys.path[0], "images/IDRiD_073.jpg")]]).launch()
