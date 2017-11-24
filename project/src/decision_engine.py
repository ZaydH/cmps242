import __main__
import tensorflow as tf


def _select_max_probability(input):

  prediction = tf.argmax(input, axis=1)
  return prediction

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