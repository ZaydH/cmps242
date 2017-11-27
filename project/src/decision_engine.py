import random

import __main__
import tensorflow as tf
import numpy as np
from const import Config


def _select_max_probability(logits):
  """
  Most naive decision engine.  Always selects the character with
  the greatest probability.

  :param logits: Output from the soft max layer
  :type logits: tf.Tensor

  :return: Index corresponding to the character selected.
  """
  return tf.argmax(logits, axis=1)


def _selected_weighted_random_probability(logits):
  # ToDo Select the weighted random probability
  tot_sum = np.sum(logits)
  cum_sum = np.cumsum(logits)
  assert(abs(tot_sum - 1) < 10^(-3)) # Since softmax, sum should be close to 1
  return int(np.searchsorted(cum_sum, random.random()))


def _selected_weighted_random_after_space(logits):
  if Config.Generate.prev_char == " ":
    return _selected_weighted_random_probability(logits)
  else:
    return _select_max_probability(logits)


def setup_decision_engine(input):
  """
  Decision Engine Setup Function

  Configures the decision engine.  In training mode, the decision
  is always the one with the maximum probability.

  In Trump mode, different options can be used on how the decision
  engine functions.

  :param input:
  :return:
  """
  if __main__.__file__ == "train.py":
    return _select_max_probability(input)