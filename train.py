# -*- coding: utf-8 -*-

"""
Train script
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: MIT

import argparse
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf

from waymo_open_dataset.metrics.python import config_util_py as config_util

from utils.data_parse import input_preprocessing
from utils.evaluation import default_metrics_config, MotionMetrics
from model.simple_model import SimpleModel


# todo: add save model path
def train_parser():
    parser = argparse.ArgumentParser(description="train arguments")
    parser.add_argument("--data_file", type=str, required=True, help='data generation yaml file needed ')

    opt = parser.parse_args()
    return opt


def train_step(inputs, model, metrics_config, motion_metrics, loss_fn, optimizer):
    """
    Execute one step of training
    :param motion_metrics: motion metric class
    :param inputs: input tensor
    :param model: trained model
    :param metrics_config: metric configuration
    :param loss_fn: loss function
    :param optimizer: optimizer
    :return:
    """
    with tf.GradientTape() as tape:
        # Collapse batch dimension and the agent per sample dimension.
        # Mask out agents that are never valid in the past.
        sample_is_valid = inputs['sample_is_valid']
        states = tf.boolean_mask(inputs['input_states'], sample_is_valid)
        gt_trajectory = tf.boolean_mask(inputs['gt_future_states'], sample_is_valid)
        gt_is_valid = tf.boolean_mask(inputs['gt_future_is_valid'], sample_is_valid)
        # Set training target.
        prediction_start = metrics_config.track_history_samples + 1
        gt_targets = gt_trajectory[:, prediction_start:, :2]
        weights = tf.cast(gt_is_valid[:, prediction_start:], tf.float32)
        pred_trajectory = model(states, training=True)
        loss_value = loss_fn(gt_targets, pred_trajectory, sample_weight=weights)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    object_type = tf.boolean_mask(inputs['object_type'], sample_is_valid)
    # Fake the score since this model does not generate any score per predicted
    # trajectory.
    pred_score = tf.ones(shape=tf.shape(pred_trajectory)[:-2])

    # Only keep `tracks_to_predict` for evaluation.
    tracks_to_predict = tf.boolean_mask(inputs['tracks_to_predict'],
                                        sample_is_valid)

    motion_metrics.update_state(
        tf.boolean_mask(pred_trajectory, tracks_to_predict),
        tf.boolean_mask(pred_score, tracks_to_predict),
        tf.boolean_mask(gt_trajectory, tracks_to_predict),
        tf.boolean_mask(gt_is_valid, tracks_to_predict),
        tf.boolean_mask(object_type, tracks_to_predict))

    return loss_value


if __name__ == '__main__':
    opt = train_parser()
    # training files
    filenames = os.listdir(opt.data_file)
    FILENAME = []
    for filename in filenames:
        FILENAME.append(os.path.join(opt.data_file, filename))

    # data loader
    dataset = tf.data.TFRecordDataset(FILENAME)
    dataset = dataset.map(input_preprocessing)
    dataset = dataset.batch(8)

    epochs = 2

    # create model
    model = SimpleModel(11, 80)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # evaluation functions
    metrics_config = default_metrics_config()
    motion_metrics = MotionMetrics(metrics_config)
    metric_names = config_util.get_breakdown_names_from_motion_config(
        metrics_config)

    for epoch in range(epochs):
        print('\nStart of epoch %d' % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, batch in enumerate(dataset):
            loss_value = train_step(batch, model, metrics_config, motion_metrics, loss_fn, optimizer)

            # Log every 10 batches.
            if step % 10 == 0:
                print('Training loss (for one batch) at step %d: %.4f' %
                      (step, float(loss_value)))
                print('Seen so far: %d samples' % ((step + 1) * 64))

        # Display metrics at the end of each epoch.
        train_metric_values = motion_metrics.result()
        for i, m in enumerate(
                ['min_ade', 'min_fde', 'miss_rate', 'overlap_rate', 'map']):
            for j, n in enumerate(metric_names):
                print('{}/{}: {}'.format(m, n, train_metric_values[i, j]))
