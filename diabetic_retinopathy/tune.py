from datetime import datetime
import logging
import gin
from absl import app, flags
import os
from ray import tune
import sys

from input_pipeline.datasets import load
from models.layers import CNN_block
from models.architectures import Architecture
from train import Trainer
from utils import utils_params, utils_misc

FLAGS = flags.FLAGS
flags.DEFINE_boolean('transfer_learning', False,
    ('Specify whether to use pretrained models for transfer learning. '
     'Otherwise, a standard CNN model will be optimized.')
)
flags.DEFINE_integer('num_samples', 4, 'Specifiy the number of trials before optimization is stopped.')
flags.DEFINE_string('search_alg', 'random_search',
    ('Specifiy the optimization algorithm to use for hyperparameter optimization. '
     'One of ("random_search", "hyperopt", "bayesopt")')
)
flags.DEFINE_string('restore_from', '',
    ('Specify a checkpoint of the optimization algorithm that it should be restored from. '
     'Only works for "hyperopt" or "bayesopt".')
)
flags.DEFINE_string('name', '', 'Specify a name for the runs, appearing as a suffix of the run folders.')
flags.DEFINE_string('metric', 'val_accuracy', 'Specify the metric to be optimized by tune. One of ("val_accuracy", "val_loss")')

def train_func(config):
    # Hyperparameters

    bindings = []

    if 'Architecture.model_type' not in config.keys():
        bindings.append('Architecture.model_type=@CNN_block')

    for key, value in config.items():
        if key == 'Architecture.model_type' and type(value) is not str:
            tf_hub_model = (
                'https://tfhub.dev/google/imagenet/inception_v1/feature_vector/4',
                'https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1',
                'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4',
            )[int(value)]
            bindings.append(f'{key}={tf_hub_model}')
        else:
            bindings.append(f'{key}={value}')

    # generate folder structures
    run_paths = utils_params.gen_run_folder('by-tune')

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    local_dir = '/absolute/path/to/diabetic_retinopathy/'

    # gin-config
    gin.parse_config_files_and_bindings([os.path.join(local_dir, 'configs', 'config.gin')], bindings)
    # look at this file to see which hyperparameters were used
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, _, ds_info = load()

    # model
    model = Architecture(input_shape=ds_info['features_shape'], n_classes=ds_info['num_classes'])

    trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
    for val_accuracy, val_loss in trainer.train():
        tune.report(val_accuracy=val_accuracy, val_loss=val_loss)


def main(argv):
    # check for dissallowed flags/flag combinations
    assert FLAGS.search_alg in ('random_search', 'bayesopt', 'hyperopt'), 'Unknown optimization algorithm'
    assert FLAGS.metric in ('val_accuracy', 'val_loss'), 'Unknown metric'
    assert not (FLAGS.search_alg=='random_search' and FLAGS.restore_from), "Random search doesn't use checkpoints"

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # set up config and search algorithm
    if FLAGS.search_alg == 'bayesopt':
        from ray.tune.suggest.bayesopt import BayesOptSearch
        bayes_space = {
            'Architecture.dropout_rate': (0, 0.9),

            'augment.min_cropping_size': (150, 200),
            'augment.max_cropping_size': (200, 256),
            'augment.cropping_probability': (0, 1),

            'augment.brightness_max_delta': (0, 0.5),

            'augment.contrast_min_factor': (0.1, 0.999), # bayes includes max value, but 1 is NOT ALLOWED (raises exception)

            'augment.hue_max_delta': (0, 0.4),

            'augment.saturation_min_factor': (0.1, 0.999), # bayes includes max value, but 1 is NOT ALLOWED (raises exception)
        }

        if FLAGS.transfer_learning:
            bayes_space.update({
                'Architecture.model_type': (0, 2.999),
            })
        else:
            bayes_space.update({
                'Architecture.base_filters': (16, 64),
                'Architecture.n_blocks': (2, 5.999), # .999 to avoid 6, but still give 5 a fair chance
                'Architecture.dense_units': (64, 1024),
            })

        search_alg = BayesOptSearch(bayes_space, metric=FLAGS.metric, mode='max' if FLAGS.metric=='val_accuracy' else 'min')

        if FLAGS.restore_from:
            search_alg.restore(FLAGS.restore_from)

        config = None

    else:
        config = {
            'Architecture.dropout_rate': tune.uniform(0, 0.9),

            'augment.min_cropping_size': tune.uniform(150, 200),
            'augment.max_cropping_size': tune.uniform(200, 256),
            'augment.cropping_probability': tune.uniform(0, 1),

            'augment.brightness_max_delta': tune.uniform(0, 0.5),

            'augment.contrast_min_factor': tune.uniform(0.1, 0.999),

            'augment.hue_max_delta': tune.uniform(0, 0.4),

            'augment.saturation_min_factor': tune.uniform(0.1, 0.999),
        }

        if FLAGS.transfer_learning:
            config.update({
                'Architecture.model_type': tune.choice([
                    'https://tfhub.dev/google/imagenet/inception_v1/feature_vector/4',
                    'https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1',
                    'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4',
                ]),
            })
        else:
            config.update({
                'Architecture.base_filters': tune.choice([16, 32, 64]),
                'Architecture.n_blocks': tune.choice([2, 3, 4, 5]),
                'Architecture.dense_units': tune.choice([64, 256, 1024, 2048]),
            })

        if FLAGS.search_alg == 'hyperopt':
            from tune.suggest.hyperopt import HyperOptSearch
            search_alg = HyperOptSearch(metric=FLAGS.metric, mode='max' if FLAGS.metric=='val_accuracy' else 'min')

            if FLAGS.restore_from:
                search_alg.restore(FLAGS.restore_from)
        else:
            search_alg = None # random search

    config_string = f'DR_{FLAGS.search_alg}_{FLAGS.metric}_{"transfer" if FLAGS.transfer_learning else "CNN"}'

    # perform hyperparameter optimization
    analysis = tune.run(
        train_func, num_samples=FLAGS.num_samples, resources_per_trial={'gpu': 1, 'cpu': 4}, search_alg=search_alg,
        name=f'tune-{config_string}_{now}{f"_{FLAGS.name}" if FLAGS.name else ""}', config=config,
    )

    local_dir = '/absolute/path/to/diabetic_retinopathy/'

    os.makedirs(os.path.join(
        local_dir,
        os.pardir,
        'tune_results',
        config_string,
        now + (f'_{FLAGS.name}' if FLAGS.name else ''),
    ), exist_ok=True)

    if search_alg is not None:
        search_alg.save(os.path.join(
            local_dir,
            os.pardir,
            'tune_results',
            config_string,
            now + (f'_{FLAGS.name}' if FLAGS.name else ''),
            f'{FLAGS.search_alg}_ckpt.pkl',
        ))

    print('Best config:', analysis.get_best_config(metric=FLAGS.metric, mode='max' if FLAGS.metric=='val_accuracy' else 'min'))

    # Get a dataframe for analyzing trial results.
    df = analysis.dataframe(metric=FLAGS.metric, mode='max' if FLAGS.metric=='val_accuracy' else 'min')
    df.to_csv(os.path.join(
        local_dir,
        os.pardir,
        'tune_results',
        config_string,
        now + (f'_{FLAGS.name}' if FLAGS.name else ''),
        'analysis.csv',
    ))

if __name__=='__main__':
    app.run(main)
