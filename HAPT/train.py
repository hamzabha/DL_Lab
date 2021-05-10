import gin
import tensorflow as tf
import logging
from glob import glob
from tensorboard.backend.event_processing import event_accumulator
import os
import numpy as np
from datetime import datetime

from evaluation.metrics import ConfusionMatrix, plot_to_image, plot_ConfusionMatrix

@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, num_classes, run_paths, total_steps, log_interval, ckpt_interval,
                 fine_tuning=False, load_checkpoint=None):

        # if log files already exist, make the starting step the last step of the previous file (when contnuing training)
        if os.path.isdir(os.path.join(run_paths['path_model_id'], 'logs', 'tensorboard', 'train')):
            latest_file = sorted(glob(os.path.join(run_paths['path_model_id'], 'logs', 'tensorboard', 'train', '*')))[-1]
            ea = event_accumulator.EventAccumulator(latest_file)
            ea.Reload()
            try:
                self.start_step = ea.Tensors('accuracy')[-1].step
            except:
                logging.fatal(
                    ("Seems like the last summary writer wasn't writter to. This might be the case "
                     "because the previous run crashed or was cancelled before the first logging step. "
                     f"To resolve this, delete the latest tensorboard events file at {latest_file} "
                     "if there is a valid older events file, otherwise simply start a new run without "
                     "continuing an existing one.")
                )
                exit()
        else:
            self.start_step = 0

        self.train_summary_writer = tf.summary.create_file_writer(
            os.path.join(run_paths['path_model_id'], 'logs', 'tensorboard', 'train')
        )
        self.test_summary_writer = tf.summary.create_file_writer(
            os.path.join(run_paths['path_model_id'], 'logs', 'tensorboard', 'test')
        )

        # Checkpoint Manager
        self.checkpoint = tf.train.Checkpoint(model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, run_paths["path_ckpts_train"], max_to_keep=None)

        if load_checkpoint:
            self.checkpoint.restore(load_checkpoint)
            logging.info(f'Training is being continued from checkpoint {load_checkpoint}')
        elif self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            logging.info(f'Training is being continued from checkpoint {self.checkpoint_manager.latest_checkpoint}')

        if fine_tuning:
            logging.info('Fine tuning: learning rate is reduced by factor 100')

        # Loss objective
        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5 if fine_tuning else 1e-3)


        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.num_classes = num_classes
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval

        self.train_confusion = ConfusionMatrix(num_classes)
        self.test_confusion = ConfusionMatrix(num_classes)

    @tf.function
    def train_step(self, windows, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(windows, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)

        # ignore all-zero labels for accuracy calculation
        sample_weight = tf.where(tf.reduce_max(labels, axis=-1) < 0.5, tf.zeros(labels.shape[:-1]), tf.ones(labels.shape[:-1]))
        self.train_accuracy(labels, predictions, sample_weight)

        self.train_confusion.update_state(labels, predictions)

    @tf.function
    def test_step(self, windows, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(windows, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)

        # ignore all-zero labels for accuracy calculation
        sample_weight = tf.where(tf.reduce_max(labels, axis=-1) < 0.5, tf.zeros(labels.shape[:-1]), tf.ones(labels.shape[:-1]))
        self.test_accuracy(labels, predictions, sample_weight)

        self.test_confusion.update_state(labels, predictions)

    def train(self):
        for idx, (images, labels) in enumerate(self.ds_train):

            step = self.start_step + idx + 1
            self.train_step(images, labels)

            if (step - self.start_step) % self.log_interval == 0:

                # Reset test metrics
                self.test_loss.reset_states()
                self.test_accuracy.reset_states()
                self.test_confusion.reset_states()

                for test_images, test_labels in self.ds_val:
                    self.test_step(test_images, test_labels)

                train_recalls, train_precisions, train_f1_scores = self.train_confusion.result()
                train_balanced_accuracy = tf.reduce_mean(train_recalls)

                test_recalls, test_precisions, test_f1_scores = self.test_confusion.result()
                test_balanced_accuracy = tf.reduce_mean(test_recalls)

                train_conf = self.train_confusion.weight.numpy().astype(np.int32)
                train_figure = plot_ConfusionMatrix(train_conf, self.num_classes)
                self.train_cm = plot_to_image(train_figure)

                test_conf = self.test_confusion.weight.numpy().astype(np.int32)
                figure = plot_ConfusionMatrix(test_conf, self.num_classes)
                self.test_cm = plot_to_image(figure)

                template = (
                    'Step {}, Loss: {:.4f}, Accuracy: {:.2f}%, Balanced Accuracy: {:.2f}%, '
                    'Test Loss: {:.4f}, Test Accuracy: {:.2f}%, Test Balanced Accuracy: {:.2f}%')

                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             train_balanced_accuracy * 100,
                                             self.test_loss.result(),
                                             self.test_accuracy.result() * 100,
                                             test_balanced_accuracy * 100))

                # Write summary to tensorboard
                with self.test_summary_writer.as_default():
                    tf.summary.scalar('loss', self.test_loss.result(), step=step)
                    tf.summary.scalar('accuracy', self.test_accuracy.result(), step=step)

                    for i, recall in enumerate(test_recalls):
                        tf.summary.scalar(f'recalls/class_{i+1}', recall, step=step)
                    for i, precision in enumerate(test_precisions):
                        tf.summary.scalar(f'precisions/class_{i+1}', precision, step=step)
                    for i, f1_score in enumerate(test_f1_scores):
                        tf.summary.scalar(f'f1_scores/class_{i+1}', f1_score, step=step)

                    tf.summary.scalar('balanced_accuracy', test_balanced_accuracy, step=step)
                    tf.summary.image('Confusion Matrix', self.test_cm, step=step)

                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss.result(), step=step)
                    tf.summary.scalar('accuracy', self.train_accuracy.result(), step=step)

                    for i, recall in enumerate(train_recalls):
                        tf.summary.scalar(f'recalls/class_{i+1}', recall, step=step)
                    for i, precision in enumerate(train_precisions):
                        tf.summary.scalar(f'precisions/class_{i+1}', precision, step=step)
                    for i, f1_score in enumerate(train_f1_scores):
                        tf.summary.scalar(f'f1_scores/class_{i+1}', f1_score, step=step)

                    tf.summary.scalar('balanced_accuracy', train_balanced_accuracy, step=step)
                    tf.summary.image('Confusion Matrix', self.train_cm, step=step)

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()
                self.train_confusion.reset_states()
                yield self.test_accuracy.result().numpy(), self.test_loss.result().numpy()

            # to prevent saving the same checkpoint twice at the end
            saved = False

            if (step - self.start_step) % self.ckpt_interval == 0:
                # Save checkpoint
                ckpt_save_path = self.checkpoint_manager.save()
                saved = True
                logging.info('Saving checkpoint for step {}: {}'.format(step, ckpt_save_path))

            if (step - self.start_step) % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                # Save final checkpoint
                if not saved:
                    ckpt_save_LastPath = self.checkpoint_manager.save()
                    logging.info('Saving last checkpoint for step {}: {}'.format(step, ckpt_save_LastPath))
                return self.test_accuracy.result().numpy(), self.test_loss.result().numpy()
