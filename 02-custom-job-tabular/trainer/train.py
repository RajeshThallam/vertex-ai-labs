

# Copyright 2021 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import hypertune
import json
import os
import re
import tensorflow as tf

from absl import app
from absl import flags
from absl import logging

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow_io import bigquery as tfio_bq

from tensorboard.plugins.hparams import api as tb_hp


FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 3, 'Nubmer of epochs')
flags.DEFINE_integer('units', 32, 'Number units in a hidden layer')
flags.DEFINE_integer('per_replica_batch_size', 128, 'Per replica batch size')
flags.DEFINE_float('dropout_ratio', 0.5, 'Dropout ratio')
flags.DEFINE_string('training_table', None, 'Training table name')
flags.DEFINE_string('validation_table', None, 'Validationa table name')
flags.mark_flag_as_required('training_table')
flags.mark_flag_as_required('validation_table')

LOCAL_MODEL_DIR = '/tmp/saved_model'
LOCAL_TB_DIR = '/tmp/logs'
LOCAL_CHECKPOINT_DIR = '/tmp/checkpoints'
EVALUATION_FILE_NAME = 'evaluations.json'

# Define features
FEATURES = {
    "tip_bin": ("categorical", tf.int64),
    "trip_month": ("categorical", tf.int64),
    "trip_day": ("categorical", tf.int64),
    "trip_day_of_week": ("categorical", tf.int64),
    "trip_hour": ("categorical", tf.int64),
    "payment_type": ("categorical", tf.string),
    "pickup_grid": ("categorical", tf.string),
    "dropoff_grid": ("categorical", tf.string),
    "euclidean": ("numeric", tf.double),
    "trip_seconds": ("numeric", tf.int64),
    "trip_miles": ("numeric", tf.double),
}
TARGET_FEATURE_NAME = 'tip_bin'

 # Set hparams for Tensorboard and Vertex hp tuner
HP_DROPOUT = tb_hp.HParam("dropout")
HP_UNITS = tb_hp.HParam("units")
HPARAMS = [
    HP_UNITS,
    HP_DROPOUT,
]
METRICS = [
    tb_hp.Metric(
        "epoch_accuracy",
        group="validation",
        display_name="epoch accuracy"),
]
HPTUNE_METRIC = 'val_accuracy'
    

def set_job_dirs():
    """Sets job directories and hyperparameter tuning trial id
    based on env variables set by Vertex AI."""
    
    model_dir = os.getenv('AIP_MODEL_DIR', LOCAL_MODEL_DIR)
    tb_dir = os.getenv('AIP_TENSORBOARD_LOG_DIR', LOCAL_TB_DIR)
    checkpoint_dir = os.getenv('AIP_CHECKPOINT_DIR', LOCAL_CHECKPOINT_DIR)
    
    path = os.path.normpath(tb_dir)
    trial_id = re.match('^[0-9]+$', path.split(os.sep)[-2])
    if not trial_id:
        trial_id = '0'
    else:
        trial_id = trial_id[0]
    logging.info(trial_id)
    
    return model_dir, tb_dir, checkpoint_dir, trial_id


def get_bq_dataset(table_name, selected_fields, target_feature='tip_bin', batch_size=32):
    
    def _transform_row(row_dict):
        trimmed_dict = {column:
                       (tf.strings.strip(tensor) if tensor.dtype == 'string' else tensor) 
                       for (column,tensor) in row_dict.items()
                       }
        target = trimmed_dict.pop(target_feature)
        return (trimmed_dict, target)

    project_id, dataset_id, table_id = table_name.split('.')
    
    client = tfio_bq.BigQueryClient()
    parent = f'projects/{project_id}'

    read_session = client.read_session(
        parent=parent,
        project_id=project_id,
        table_id=table_id,
        dataset_id=dataset_id,
        selected_fields=selected_fields,
    )

    dataset = read_session.parallel_read_rows().map(_transform_row).batch(batch_size)
    
    return dataset


def get_category_encoding_layer(name, dataset, dtype):
    """Creates a CategoryEncoding layer for a given feature."""

    if dtype == tf.string:
      index = preprocessing.StringLookup()
    else:
      index = preprocessing.IntegerLookup()

    feature_ds = dataset.map(lambda x, y: x[name])
    index.adapt(feature_ds)
    encoder = preprocessing.CategoryEncoding(max_tokens=index.vocab_size())

    return lambda feature: encoder(index(feature))


def get_normalization_layer(name, dataset):
  """"Creates a Normalization layer for a given feature."""
  normalizer = preprocessing.Normalization()

  feature_ds = dataset.map(lambda x, y: x[name])
  normalizer.adapt(feature_ds)

  return normalizer


def create_model(dataset, input_features, units, dropout_ratio):
    """Creates a binary classifier for Chicago Taxi tip prediction task."""
    
    all_inputs = []
    encoded_features = []
    for feature_name, feature_info in input_features.items():
        col = tf.keras.Input(shape=(1,), name=feature_name, dtype=feature_info[1])
        if feature_info[0] == 'categorical':
            
            encoding_layer = get_category_encoding_layer(feature_name, 
                                                         dataset,
                                                         feature_info[1])
        else:
            encoding_layer = get_normalization_layer(feature_name,
                                                     dataset) 
        encoded_col = encoding_layer(col)
        all_inputs.append(col)
        encoded_features.append(encoded_col)
        
    all_features = tf.keras.layers.concatenate(encoded_features)
    
    x = tf.keras.layers.Dense(units, activation="relu")(all_features)
    x = tf.keras.layers.Dropout(dropout_ratio)(x)
    output = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(all_inputs, output)
    
    return model


class HptuneCallback(tf.keras.callbacks.Callback):
    """
    A custom Keras callback class that reports a metric to hypertuner
    at the end of each epoch.
    """
    
    def __init__(self, metric_tag, metric_value):
        super(HptuneCallback, self).__init__()
        self.metric_tag = metric_tag
        self.metric_value = metric_value
        self.hpt = hypertune.HyperTune()
        
    def on_epoch_end(self, epoch, logs=None):
        self.hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag=self.metric_tag,
            metric_value=logs[self.metric_value],
            global_step=epoch)
        

def main(argv):
    del argv
    
    # Set distribution strategy
    strategy = tf.distribute.MirroredStrategy()
    
    global_batch_size = (strategy.num_replicas_in_sync *
                         FLAGS.per_replica_batch_size)
    
    # Prepare datasets
    selected_fields = {key: {'output_type': value[1]} for key, value in FEATURES.items()}
    validation_ds = get_bq_dataset(FLAGS.validation_table, 
                                   selected_fields, 
                                   batch_size=global_batch_size)
    training_ds = get_bq_dataset(FLAGS.training_table,
                                 selected_fields,
                                 batch_size=global_batch_size)
    
    # Configure Tensorboard hparams
    model_dir, tb_dir, checkpoint_dir, trial_id = set_job_dirs()
    with tf.summary.create_file_writer(tb_dir).as_default():
        tb_hp.hparams_config(hparams=HPARAMS, metrics=METRICS)
        
    hparams = {
        HP_UNITS: FLAGS.units,
        HP_DROPOUT: FLAGS.dropout_ratio
    }
    
    # Create the model
    input_features = {key: value for key, value in FEATURES.items() if key != TARGET_FEATURE_NAME}
    logging.info('Creating the model ...')
    with strategy.scope():
        model = create_model(training_ds, input_features, hparams[HP_UNITS], hparams[HP_DROPOUT])
        model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    # Configure training regimen
    callbacks = [tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=checkpoint_dir)]
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=tb_dir, 
                                                    update_freq='batch',
                                                    profile_batch=0))
    callbacks.append(tb_hp.KerasCallback(writer=tb_dir, 
                                         hparams=hparams,
                                         trial_id=trial_id))
    callbacks.append(HptuneCallback(HPTUNE_METRIC, HPTUNE_METRIC))
    
    # Start training
    logging.info('Starting training ...')
    model.fit(training_ds, 
              epochs=FLAGS.epochs, 
              validation_data=validation_ds,
              callbacks=callbacks)
    
    # Save trained model
    logging.info('Training completed. Saving the trained model to: {}'.format(model_dir))
    model.save(model_dir)  
    
    
if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
