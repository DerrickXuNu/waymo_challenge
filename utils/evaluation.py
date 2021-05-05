# -*- coding: utf-8 -*-

"""
Evaluation functions for motion prediction
"""
# Author: Waymo Open Dataset, Runsheng Xu <rxx3386@ucla.edu>
# License: MIT

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2


def default_metrics_config():
    config = motion_metrics_pb2.MotionMetricsConfig()
    config_text = """
  track_steps_per_second: 10
  prediction_steps_per_second: 2
  track_history_samples: 10
  track_future_samples: 80
  speed_lower_bound: 1.4
  speed_upper_bound: 11.0
  speed_scale_lower: 0.5
  speed_scale_upper: 1.0
  step_configurations {
    measurement_step: 5
    lateral_miss_threshold: 1.0
    longitudinal_miss_threshold: 2.0
  }
  step_configurations {
    measurement_step: 9
    lateral_miss_threshold: 1.8
    longitudinal_miss_threshold: 3.6
  }
  step_configurations {
    measurement_step: 15
    lateral_miss_threshold: 3.0
    longitudinal_miss_threshold: 6.0
  }
  max_predictions: 6
  """
    text_format.Parse(config_text, config)
    return config


class MotionMetrics(tf.keras.metrics.Metric):
    """Wrapper for motion metrics computation."""
    def __init__(self, config):
        super().__init__()
        self._ground_truth_trajectory = []
        self._ground_truth_is_valid = []
        self._prediction_trajectory = []
        self._prediction_score = []
        self._object_type = []
        self._metrics_config = config

    def reset_state(self):
        self._ground_truth_trajectory = []
        self._ground_truth_is_valid = []
        self._prediction_trajectory = []
        self._prediction_score = []
        self._object_type = []

    def update_state(self, prediction_trajectory, prediction_score,
                     ground_truth_trajectory, ground_truth_is_valid, object_type):
        self._prediction_trajectory.append(prediction_trajectory)
        self._prediction_score.append(prediction_score)
        self._ground_truth_trajectory.append(ground_truth_trajectory)
        self._ground_truth_is_valid.append(ground_truth_is_valid)
        self._object_type.append(object_type)

    def result(self):
        # [batch_size, steps, 2].
        prediction_trajectory = tf.concat(self._prediction_trajectory, 0)
        # [batch_size].
        prediction_score = tf.concat(self._prediction_score, 0)
        # [batch_size, gt_steps, 7].
        ground_truth_trajectory = tf.concat(self._ground_truth_trajectory, 0)
        # [batch_size, gt_steps].
        ground_truth_is_valid = tf.concat(self._ground_truth_is_valid, 0)
        # [batch_size].
        object_type = tf.cast(tf.concat(self._object_type, 0), tf.int64)

        # We are predicting more steps than needed by the eval code. Subsample.
        interval = (
                self._metrics_config.track_steps_per_second //
                self._metrics_config.prediction_steps_per_second)
        prediction_trajectory = prediction_trajectory[:, (interval - 1)::interval]

        # Prepare these into shapes expected by the metrics computation.
        # TODO: This is only for uni-model, multi-modal needs change in the future
        # [batch_size, top_k, num_agents_per_joint_prediction, pred_steps, 2].
        # top_k is 1 because we have a uni-modal model.
        # num_agents_per_joint_prediction is also 1 here.
        prediction_trajectory = prediction_trajectory[:, tf.newaxis, tf.newaxis]
        # [batch_size, top_k].
        prediction_score = prediction_score[:, tf.newaxis]
        # [batch_size, num_agents_per_joint_prediction, gt_steps, 7].
        ground_truth_trajectory = ground_truth_trajectory[:, tf.newaxis]
        # [batch_size, num_agents_per_joint_prediction, gt_steps].
        ground_truth_is_valid = ground_truth_is_valid[:, tf.newaxis]
        # [batch_size, num_agents_per_joint_prediction].
        object_type = object_type[:, tf.newaxis]

        return py_metrics_ops.motion_metrics(
            config=self._metrics_config.SerializeToString(),
            prediction_trajectory=prediction_trajectory,
            prediction_score=prediction_score,
            ground_truth_trajectory=ground_truth_trajectory,
            ground_truth_is_valid=ground_truth_is_valid,
            object_type=object_type)
