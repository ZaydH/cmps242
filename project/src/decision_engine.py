import __main__
import tensorflow as tf
from enum import Enum


class DecisionFunction(Enum):
  ArgMax = 0


def _select_max_probability(logits):
  """
  Most naive decision engine.  Always selects the character with
  the greatest probability.

  :param logits: Output from the soft max layer
  :type logits: tf.Tensor

  :return: Index corresponding to the character selected.
  """
  return tf.argmax(logits, axis=1)


def _selected_weighted_random_probability(input):
  # ToDo Select the weighted random probability
  pass


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