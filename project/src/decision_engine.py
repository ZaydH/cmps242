import random

import __main__
import tensorflow as tf
import numpy as np
from const import Config


def select_max_probability(sess, softmax_out):
  """
  Most naive decision engine.  Always selects the character with
  the greatest probability.

  :param softmax_out: Output from the soft max layer
  :type softmax_out: tf.Tensor

  :return: Index corresponding to the character selected.
  :rtype: int
  """
  return sess.run(tf.argmax(softmax_out, 0))


def _selected_weighted_random_probability(_, softmax_out):
  """
  Performs a weighted random selection where the likelihood a particular
  character is selected is proportional to the softmax output of that
  character.

  :param _: Ignored parameter for consistency with other decision functions

  :param softmax_out: Output from the soft max layer
  :type softmax_out: tf.Tensor

  :return: Index corresponding to the selected character
  :rtype: int
  """
  tot_sum = np.sum(softmax_out)
  assert abs(tot_sum - 1) < 10 ** (-3)  # Since softmax, sum should be close to 1

  cum_sum = np.cumsum(softmax_out)
  return int(np.searchsorted(cum_sum, random.random()))


def selected_weighted_random_after_space(sess, logits):
  if Config.Generate.prev_char == " ":
    return _selected_weighted_random_probability(sess, logits)
  else:
    return select_max_probability(sess, logits)


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
    return select_max_probability(input)
