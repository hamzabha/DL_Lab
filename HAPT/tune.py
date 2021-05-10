from datetime import datetime
import logging
import gin
from absl import app, flags
import os
from ray import tune
import sys

from input_pipeline.datasets import load
from models.architectures import Architecture
from train import Trainer
from utils import utils_params, utils_misc

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_samples', 4, 'Specifiy the number of trials before optimization is stopped.')
flags.DEFINE_string('search_alg', 'bayesopt',
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

    for key, value in config.items():
        if key == 'Architecture.rnn_block_type':
            rnn_block_type = [
                '"unidirectional"',
                '"bidirectional"',
                '"cascaded"',
                '"gru"',
            ][int(value)]
            bindings.append(f'{key}={rnn_block_type}')
        else:
            bindings.append(f'{key}={value}')

    # generate folder structures
    run_paths = utils_params.gen_run_folder('by-tune')

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    local_dir = '/absolute/path/to/HAPT/'

    # gin-config
    gin.parse_config_files_and_bindings([os.path.join(local_dir, 'configs', 'config.gin')], bindings)
    # look at this file to see which hyperparameters were used
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, _, ds_info = load()

    # model
    model = Architecture(features=ds_info['num_features'], n_classes=ds_info['num_classes'])

    trainer = Trainer(model, ds_train, ds_val, ds_info['num_classes'], run_paths)
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
            'Architecture.rnn_block_type': (0, 3.999),
            'Architecture.rnn_units': (16, 256),
            'Architecture.num_rnn_layers': (1, 3.999),
            'Architecture.dense_units': (32, 1024),
            'Architecture.dropout_rate': (0, 0.9),
        }

        search_alg = BayesOptSearch(bayes_space, metric=FLAGS.metric, mode='max' if FLAGS.metric=='val_accuracy' else 'min')

        if FLAGS.restore_from:
            search_alg.restore(FLAGS.restore_from)

        config = None

    else:
        config = {
            'Architecture.rnn_block_type': tune.choice([0, 1, 2, 3]),
            'Architecture.rnn_units': tune.choice([16, 32, 64, 128, 256]),
            'Architecture.num_rnn_layers': tune.choice([0, 1, 2, 3, 4]),
            'Architecture.dense_units': tune.choice([32, 128, 512, 1024]),
            'Architecture.dropout_rate': tune.uniform(0, 0.9),
        }

        if FLAGS.search_alg == 'hyperopt':
            from tune.suggest.hyperopt import HyperOptSearch
            search_alg = HyperOptSearch(metric=FLAGS.metric, mode='max' if FLAGS.metric=='val_accuracy' else 'min')

            if FLAGS.restore_from:
                search_alg.restore(FLAGS.restore_from)
        else:
            search_alg = None # random search

    config_string = f'HAR_{FLAGS.search_alg}_{FLAGS.metric}'

    # perform hyperparameter optimization
    analysis = tune.run(
        train_func, num_samples=FLAGS.num_samples, resources_per_trial={'gpu': 1, 'cpu': 1}, search_alg=search_alg,
        name=f'tune-{config_string}_{now}{f"_{FLAGS.name}" if FLAGS.name else ""}', config=config,
    )

    local_dir = '/absolute/path/to/HAPT/'

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
