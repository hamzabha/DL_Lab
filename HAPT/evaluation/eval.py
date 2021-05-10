import tensorflow as tf
from .metrics import ConfusionMatrix
import os
import numpy as np
import matplotlib.pyplot as plt

def evaluate(model, checkpoint_or_folder, ds_test, ds_info, run_paths):

    checkpoint = tf.train.Checkpoint(model=model)
    if os.path.isdir(checkpoint_or_folder):
        checkpoint.restore(
            tf.train.CheckpointManager(checkpoint, run_paths["path_ckpts_train"], max_to_keep=None).latest_checkpoint
        )
    else:
        checkpoint.restore(checkpoint_or_folder)

    loss_obj = tf.keras.losses.CategoricalCrossentropy()

    loss_metric = tf.keras.metrics.Mean(name='test_loss')
    accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    confusion = ConfusionMatrix(ds_info['num_classes'])

    for test_images, test_labels in ds_test:
        test_step(model, test_images, test_labels, loss_obj, loss_metric, accuracy, confusion)
    conf = confusion.weight.numpy().astype(np.int32)
    figure = image(conf, ds_info['num_classes'])
    plt.close(figure)
    recalls, _, _ = confusion.result()
    return loss_metric.result(), accuracy.result(), tf.reduce_mean(recalls), figure

@tf.function
def test_step(model, images, labels, loss_obj, loss_metric, accuracy, confusion):
    predictions = model(images, training=False)
    t_loss = loss_obj(labels, predictions)

    loss_metric(t_loss)

    # ignore all-zero labels for accuracy calculation
    sample_weight = tf.where(tf.reduce_max(labels, axis=-1) < 0.5, tf.zeros(labels.shape[:-1]), tf.ones(labels.shape[:-1]))
    accuracy(labels, predictions, sample_weight)
    confusion.update_state(labels, predictions)

def image(Confmat, num_classes):
    lbl = {0: "Walking",
           1: "Walking_upstairs",
           2: "Walking_downstairs",
           3: "Sitting",
           4: "Standing",
           5: "Laying",
           6: "Stand_to_Sit",
           7: "Sit_to_Stand",
           8: "Sit_to_Lie",
           9: "Lie_to_Sit",
           10: "Stand_to_Lie",
           11: "Lie_to_stand"}
    lbl = {int(k): v for k, v in lbl.items()}

    Confmat = np.around(Confmat.astype('float') / (Confmat.sum(axis=1))[:, np.newaxis], decimals=2)
    figure, ax = plt.subplots(figsize=(11, 11))
    plt.imshow(Confmat, cmap=plt.cm.Reds)
    for i in range(num_classes):
        for j in range(num_classes):
            n = Confmat[j, i]
            ax.text(i, j, str(n), va='center', ha='center', size=10, fontdict=None)
    plt.xticks(range(num_classes))
    plt.yticks(range(num_classes), ['{} ({})'.format(lbl[i], i) for i in range(num_classes)])
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.show()
    return figure
