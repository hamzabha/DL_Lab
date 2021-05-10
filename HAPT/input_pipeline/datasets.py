import numpy as np
import pandas as pd
from glob import glob
import gin
import os
from input_pipeline.preprocessing import normalization, windowing
import tensorflow as tf
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
data_dir = '/home/data/HAPT_dataset'

@gin.configurable
def load(data_dir, for_visualization=False, only_test_set=False):
    # defining the paths for the accelerometer, gyroscope and labels files
    data_path = sorted(glob(os.path.join(data_dir, 'RawData', '*')))
    acc_path = sorted(glob(os.path.join(data_dir, 'RawData', 'acc*.txt')))
    gyro_path = sorted(glob(os.path.join(data_dir, 'RawData', 'gyro*.txt')))
    labels_file_path = data_path[-1]

    acc_columns = ['acc_x', 'acc_y', 'acc_z']
    gyro_columns = ['gyro_x', 'gyro_y', 'gyro_z']
    column_names = ['exp_id', 'user_id', 'activity', 'start_pt', 'end_pt']
    labels = pd.read_csv(labels_file_path, sep=" ", header=None, names=column_names)
    data = []

    for acc, gyro in zip(acc_path, gyro_path):
        exp_id = int(acc.split("exp")[1][:2])
        user_id = int(acc.split("user")[1][:2])

        acc_data = pd.read_table(acc, sep=" ", header=None, names=acc_columns)
        gyro_data = pd.read_table(gyro, sep=" ", header=None, names=gyro_columns)

        temp_label = labels[
            (labels.exp_id == exp_id) & (labels.user_id == user_id)
            ]

        acc_data['label'] = pd.Series([-1 for x in range(len(acc_data.index))], index=acc_data.index)
        acc_data['exp_id'] = pd.Series([0 for x in range(len(acc_data.index))], index=acc_data.index)
        gyro_data['exp_id'] = pd.Series([0 for x in range(len(gyro_data.index))], index=gyro_data.index)
        acc_data['user'] = pd.Series([0 for x in range(len(acc_data.index))], index=acc_data.index)

        for exp_id, user_id, act_id, start, end in temp_label.values:
            acc_data.loc[:, 'exp_id'] = exp_id
            gyro_data.loc[:, 'exp_id'] = exp_id
            acc_data.loc[:, 'user'] = user_id
            acc_data.loc[start:end+1, 'label'] = act_id-1 # "-1" to change classes 1-12 to 0-11
            acc_data = acc_data[['exp_id', 'user', 'label', 'acc_x', 'acc_y', 'acc_z']]
            acc_data[['acc_x', 'acc_y', 'acc_z']] = normalization(acc_data[['acc_x', 'acc_y', 'acc_z']])
            gyro_data = normalization(gyro_data[['gyro_x', 'gyro_y', 'gyro_z']])


        data_norm = pd.concat([acc_data, gyro_data], axis=1)
        data.append(data_norm)

    df = pd.concat(data, axis=0).reset_index()
    df = df.drop('index', axis=1)
    df = df[['exp_id', 'user', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'label']]

    unique_labels = set(df['label'])
    num_classes = len(unique_labels)
    if -1 in unique_labels:
        num_classes -= 1

    # separating the files into train, validation and test dataframes
    df_train = df[df['user'] < 22]
    df_val = df[df['user'] > 27]
    df_test = df[(df['user'] > 21) & (df['user'] < 28)]

    # dropping the 'exp_id' and the 'user' columns
    df_train = df_train[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'label']]
    df_val = df_val[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'label']]
    df_test = df_test[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'label']]

    ds_info = {
        'num_training_samples': len(df_train),
        'num_test_samples': len(df_test),
        'num_classes': num_classes,
        'num_features': (len(df_train.columns) - 1)
    }

    if only_test_set:
        return windowing(df_test, buffer_size=len(df_test) // 10, for_visualization=for_visualization), ds_info

    ds_train = windowing(df_train, buffer_size=ds_info['num_training_samples'] // 10, repeat=True, shuffle=True, remove_unlabelled=True, for_visualization=for_visualization)
    ds_val = windowing(df_val, buffer_size=ds_info['num_training_samples'] // 10, for_visualization=for_visualization)
    ds_test = windowing(df_test, buffer_size=ds_info['num_training_samples'] // 10, for_visualization=for_visualization)

    return ds_train, ds_val, ds_test, ds_info