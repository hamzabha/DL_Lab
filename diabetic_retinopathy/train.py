import gin
import tensorflow as tf
import logging
from glob import glob
from tensorboard.backend.event_processing import event_accumulator
import os
import numpy as np

from evaluation.metrics import ConfusionMatrix, plot_to_image, plot_ConfusionMatrix

@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, ds_info, run_paths, total_steps, log_interval, ckpt_interval, fine_tuning=False, load_checkpoint=None):
        
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
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5 if fine_tuning else 1e-3)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval

        self.train_confusion = ConfusionMatrix(ds_info['num_classes'])
        self.test_confusion = ConfusionMatrix(ds_info['num_classes'])

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)
        self.train_confusion.update_state(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)
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


                self.train_recall, self.train_precision, self.train_f1_score, self.train_balanced_accuracy = self.train_confusion.result()
                self.test_recall, self.test_precision, self.test_f1_score, self.test_balanced_accuracy = self.test_confusion.result()

                train_conf = self.train_confusion.weight.numpy().astype(np.int32)
                train_figure = plot_ConfusionMatrix(train_conf, 2)
                self.train_cm = plot_to_image(train_figure)

                test_conf = self.test_confusion.weight.numpy().astype(np.int32)
                figure = plot_ConfusionMatrix(test_conf, 2)
                self.test_cm = plot_to_image(figure)

                template = ('Step {}, Loss: {:.4f}, Accuracy: {:.2f}%, Recall: {:.2f}%, Precision: {:.2f}%, F1 Score: {:.2f}%, Balanced Accuracy: {:.2f}%, '
                            'Test Loss: {:.4f}, Test Accuracy: {:.2f}%, Test Recall: {:.2f}%, Test Precision: {:.2f}%, Test F1 Score: {:.2f}%, Test Balanced Accuracy: {:.2f}%')


                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.train_recall * 100,
                                             self.train_precision * 100,
                                             self.train_f1_score * 100,
                                             self.train_balanced_accuracy * 100,
                                             self.test_loss.result(),
                                             self.test_accuracy.result() * 100,
                                             self.test_recall * 100,
                                             self.test_precision * 100,
                                             self.test_f1_score * 100,
                                             self.test_balanced_accuracy * 100))

                # Write summary to tensorboard
                with self.test_summary_writer.as_default():
                    tf.summary.scalar('loss', self.test_loss.result(), step=step)
                    tf.summary.scalar('accuracy', self.test_accuracy.result(), step=step)
                    tf.summary.scalar('recall', self.test_recall, step=step)
                    tf.summary.scalar('precision', self.test_precision, step=step)
                    tf.summary.scalar('f1_score', self.test_f1_score, step=step)
                    tf.summary.scalar('Balanced accuracy', self.test_balanced_accuracy, step=step)
                    tf.summary.image('Confusion Matrix', self.test_cm, step=step)

                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss.result(), step=step)
                    tf.summary.scalar('accuracy', self.train_accuracy.result(), step=step)
                    tf.summary.scalar('recall', self.train_recall, step=step)
                    tf.summary.scalar('precision', self.train_precision, step=step)
                    tf.summary.scalar('f1_score', self.train_f1_score, step=step)
                    tf.summary.scalar('Balanced accuracy', self.train_balanced_accuracy, step=step)
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

            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                # Save final checkpoint
                if not saved:
                    ckpt_save_LastPath = self.checkpoint_manager.save()
                    logging.info('Saving last checkpoint for step {}: {}'.format(step, ckpt_save_LastPath))
                return self.test_accuracy.result().numpy(), self.test_loss.result().numpy()
