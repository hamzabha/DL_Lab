import gin
import logging
from absl import app, flags
import os

from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import Architecture
from models.layers import CNN_block


FLAGS = flags.FLAGS
flags.DEFINE_string('name', '',
    ('Specify the name of a new training run, appearing as a suffix of the run folder. '
     'Is only allowed if a new training run is being started '
     '(neither --evaluate_from nor --continue_training_from is specified).')
)
flags.DEFINE_string('continue_training_from', '',
    ('Specify either the exact checkpoint inside a run folder, or the run folder itself to use its latest checkpoint '
     'to continue training the model saved there. configs/config.gin will then be ignored, and the gin operative '
     'saved in the run folder will be used instead, unless --force_use_config is specified.')
)
flags.DEFINE_string('evaluate_from', '',
    ('Specify either the exact checkpoint inside a run folder, or the run folder itself to use its latest checkpoint for evaluation. '
     'configs/config.gin will then be ignored, and the gin operative saved in the run folder will be used instead.')
)
flags.DEFINE_boolean('force_use_config', False,
    ('Specify whether to use the current config at configs/config.gin when training is continued from an existing run folder '
     'even though the original config used is saved there. Note that changes to the architecture will not work. '
     'Use at own risk.')
)
flags.DEFINE_boolean('fine_tune', False,
    ('Specify whether to fine tune a model. This is only allowed if training is being continued from an existing checkpoint '
     'using --continue_training_from. This will reduce the learning rate by the factor 100. In case of a transfer learning model from tf-hub, '
     'another effect is that the base network will also be made trainable.')
)

def main(argv):
    # check for disallowed flag combinations
    assert not (FLAGS.continue_training_from and FLAGS.evaluate_from), \
        "Only one of --continue_training_from and --evaluate_from can be specified at once."
    assert not (FLAGS.name and (FLAGS.continue_training_from or FLAGS.evaluate_from)), \
        "Can't set the name of an existing folder being reused for continued training or evaluation."
    assert not (FLAGS.force_use_config and not FLAGS.continue_training_from ), \
        "--force_use_config is only allowed when '--continue_training_from' is specified"
    assert not (FLAGS.fine_tune and not FLAGS.continue_training_from), \
        "--fine_tune is only allowed when continuing training from a checkpoint using --continue_training_from"

    load_checkpoint = None
    if not FLAGS.evaluate_from:
        # Training

        # implementation detail needed to avoid accidentally reusing an existing folder whose name coincidentally is equal to `FLAGS.name`
        force_create_new = os.path.isdir(FLAGS.name)

        continue_folder = FLAGS.continue_training_from
        # if a checkpoint has been specified, remember to use that checkpoint and get the run folder (2 directories above)
        if FLAGS.continue_training_from and not os.path.isdir(FLAGS.continue_training_from):
            load_checkpoint = FLAGS.continue_training_from
            continue_folder = os.path.join(os.path.dirname(FLAGS.continue_training_from), os.pardir)

        # generate (or reuse existing) folder structures
        run_paths = utils_params.gen_run_folder(continue_folder or FLAGS.name, force_create_new=force_create_new)
        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        if FLAGS.continue_training_from:
            # An existing folder was specified: Use existing config_operative.gin, unless force_use_config is specified
            if FLAGS.force_use_config:
                gin.parse_config_files_and_bindings([os.path.join(os.path.dirname(__file__), 'configs', 'config.gin')], [])
            else:
                gin.parse_config_files_and_bindings([run_paths['path_gin']], [])
        else:
            # Nothing specified: New folder was created, so use and save configs/config.gin
            gin.parse_config_files_and_bindings([os.path.join(os.path.dirname(__file__), 'configs', 'config.gin')], [])
            utils_params.save_config(run_paths['path_gin'], gin.config_str())

    else:
        # Evaluation
        run_folder = FLAGS.evaluate_from if os.path.isdir(FLAGS.evaluate_from) else os.path.join(os.path.dirname(FLAGS.evaluate_from), os.pardir)

        # reuse existing folder structures
        run_paths = utils_params.gen_run_folder(run_folder)
        # set loggers (appending to existing log file)
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
        # reuse the saved gin operative of the run folder as config
        gin.parse_config_files_and_bindings([run_paths['path_gin']], [])

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load()

    # model
    model = Architecture(input_shape=ds_info['features_shape'], n_classes=ds_info['num_classes'], fine_tuning=FLAGS.fine_tune)

    if not FLAGS.evaluate_from:
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths, fine_tuning=FLAGS.fine_tune, load_checkpoint=load_checkpoint)
        for _ in trainer.train():
            continue
    else:
        loss, accuracy, recall, precision, f1_score, balanced_accuracy, _ = evaluate(model,
                                                                         FLAGS.evaluate_from,
                                                                         ds_test,
                                                                         ds_info,
                                                                         run_paths)

        logging.info(f'Evaluation results from {"latest checkpoint" if os.path.isdir(FLAGS.evaluate_from) else FLAGS.evaluate_from}:')
        template = 'Loss: {}, Accuracy: {}%, Recall: {}%, Precision: {}%, f1_score: {}%, Balanced Accuracy: {}%'
        logging.info(template.format(loss,
                                     accuracy * 100,
                                     recall * 100,
                                     precision * 100,
                                     f1_score * 100,
                                     balanced_accuracy * 100))

if __name__ == "__main__":
    app.run(main)
