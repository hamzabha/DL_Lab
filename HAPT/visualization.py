import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import gin
import logging
from absl import app, flags
import os

from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import Architecture


FLAGS = flags.FLAGS
flags.DEFINE_string('visualize_from', '',
    ('Specify either the exact checkpoint inside a run folder, or the run folder itself to use its latest checkpoint for evaluation. '
     'configs/config.gin will then be ignored, and the gin operative saved in the run folder will be used instead.')
)

def main(argv):
    # check for disallowed flag combinations
    assert FLAGS.visualize_from, 'Need to specify a model to use for visualization'
    
    run_folder = FLAGS.visualize_from if os.path.isdir(FLAGS.visualize_from) else os.path.join(os.path.dirname(FLAGS.visualize_from), os.pardir)

    # reuse existing folder structures
    run_paths = utils_params.gen_run_folder(run_folder)
    # set loggers (appending to existing log file)
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
    # reuse the saved gin operative of the run folder as config
    gin.parse_config_files_and_bindings([run_paths['path_gin']], [])

    # setup pipeline
    ds_test, ds_info = datasets.load(for_visualization=True, only_test_set=True)

    # model
    model = Architecture(features=ds_info['num_features'], n_classes=ds_info['num_classes'])

    checkpoint = tf.train.Checkpoint(model=model)
    if os.path.isdir(FLAGS.visualize_from):
        checkpoint.restore(
            tf.train.CheckpointManager(checkpoint, run_paths["path_ckpts_train"], max_to_keep=None).latest_checkpoint
        )
    else:
        checkpoint.restore(FLAGS.visualize_from)

    for windows, tf_labels in ds_test:
        
        acc_x = tf.reshape(windows[:, :, 0], [-1])
        acc_y = tf.reshape(windows[:, :, 1], [-1])
        acc_z = tf.reshape(windows[:, :, 2], [-1])
        gyro_x = tf.reshape(windows[:, :, 3], [-1])
        gyro_y = tf.reshape(windows[:, :, 4], [-1])
        gyro_z = tf.reshape(windows[:, :, 5], [-1])

        true_labels = tf.argmax(tf_labels, axis=-1)

        # where the label was all-zero, fill in a -1 instead of the argmax
        true_labels = tf.where(tf.reduce_max(tf_labels, axis=-1) < 0.5, tf.zeros_like(true_labels)-1, true_labels)

        true_labels = tf.reshape(true_labels, [-1])
        break

    model_out = model(windows, training=False)
    predicted_labels = tf.reshape(tf.argmax(model_out, axis=-1), [-1])

    visualize(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, predicted_labels, true_labels)


def visualize(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, labels, true_labels=None,
    colors=[
        '#FFB300',
        '#803E75',
        '#FF6800',
        '#A6BDD7',
        '#C10020',
        '#CEA262',
        '#817066',
        '#007D34',
        '#F6768E',
        '#00538A',
        '#FF7A5C',
        '#53377A',
        '#FFFFFF', # so that -1 (unlabelled) maps to white
    ]):

    time_axis = np.arange(acc_x.shape[0]) / 50.0

    f, (a0, a1, a2) = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [10, 10, 1]})

    # plot acceleration data
    a0.plot(time_axis, acc_x, label="X-Direction")
    a0.plot(time_axis, acc_y, label="Y-Direction")
    a0.plot(time_axis, acc_z, label="Z-Direction")
    make_background(a0, labels, colors, true_labels=true_labels)

    a0.set_xlim((0, time_axis[-1]))
    minval = min(np.min(acc_x), np.min(acc_y), np.min(acc_z))
    maxval = max(np.max(acc_x), np.max(acc_y), np.max(acc_z))
    valrange = maxval - minval
    a0.set_ylim((minval - valrange/4 - valrange*0.05, maxval + valrange*0.05))
    a0.set_xlabel('Time in s')
    a0.set_ylabel('Normalized Acceleration')
    a0.set_title('Accelorometer')

    a0.legend(loc='upper left', ncol=3)

    # plot gyroscope data
    a1.plot(time_axis, gyro_x, label="Around X-Axis")
    a1.plot(time_axis, gyro_y, label="Around Y-Axis")
    a1.plot(time_axis, gyro_z, label="Around Z-Axis")
    make_background(a1, labels, colors, true_labels=true_labels)

    a1.set_xlim((0, time_axis[-1]))
    minval = min(np.min(gyro_x), np.min(gyro_y), np.min(gyro_z))
    maxval = max(np.max(gyro_x), np.max(gyro_y), np.max(gyro_z))
    valrange = maxval - minval
    a1.set_ylim((minval - valrange/4 - valrange*0.05, maxval + valrange*0.05))
    a1.set_xlabel('Time in s')
    a1.set_ylabel('Normalized Rotational Speed')
    a1.set_title('Gyroscope')

    a1.legend(loc='upper left', ncol=3)

    # create the legend showing which color refers to which activity
    make_legend(a2, colors)

    f.tight_layout()
    plt.show()


def make_legend(ax, colors):
    labels = [
        'Walking',
        'Walking upstairs',
        'Walking downstairs',
        'Sitting',
        'Standing',
        'Laying',
        'Stand to Sit',
        'Sit to Stand',
        'Sit to Lie',
        'Lie to Sit',
        'Stand to Lie',
        'Lie to Stand',
    ]
    ax.bar(labels, np.ones(12), 1, color=colors)
    plt.xticks(rotation=45)
    plt.gca().get_yaxis().set_visible(False)
    ax.set_xticklabels(labels, ha='right')

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def make_background(axes, labels, colors, true_labels=None):
    def draw_labels(labs, ymin=0, ymax=1):
        last_label = labs[0]
        start = 0
        for idx, label in enumerate(labs):
            if label != last_label or idx == len(labs)-1:
                axes.axvspan(start/50, idx/50, ymin, ymax, facecolor=colors[int(last_label)], zorder=-100)
                start = idx
            last_label = label

    draw_labels(labels, 0, 0.1)
    x_margin = 0.0075
    axes.text(x_margin, 0.05,'Prediction',
        horizontalalignment='left',
        verticalalignment='center',
        transform=axes.transAxes,
        fontweight='bold'
    )
    if true_labels is not None:
        draw_labels(true_labels, 0.1, 0.2)
        axes.text(x_margin, 0.15,'Ground Truth',
            horizontalalignment='left',
            verticalalignment='center',
            transform=axes.transAxes,
            fontweight='bold'
        )


if __name__ == "__main__":
    app.run(main)