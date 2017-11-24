import tensorflow as tf

import const


def _get_tf_rand_uniform(shape):
    """
    TensorFlow Random Uniform Variable Generator

    Initialize a random set of weights uniformly at random
    between -1 and 1.

    :param shape: Shape of the random object to generate
    :type shape: tf.shape

    :return: Uniform random TensorFlow variable of the specified shape
    :rtype: tf.Variable
    """
    obj = tf.random_uniform(shape, minval=-1., maxval=1.)
    return tf.Variable(obj)


def _get_tf_rand_normal(shape):
    """
    Random Normal Variable Generator

    Generates a random object of the specified shape that is
    using the normal distribution with mean 0 and standard deviation 1.

    :param shape: Shape of the random object to generate
    :type shape: tf.shape

    :return: Normal random TensorFlow variable of the specified shape
    :rtype: tf.Variable
    """
    obj = tf.random_normal(shape, mean=0., stddev=1.)
    return tf.Variable(obj)


def _build_feed_forward(input, vocab_size, rand_func):
  """
  Feed-Forward Network Builder

  Constructs and initializes the feed-forward network.

  :param input: Input to the feed-forward network.
  :type input: tf.Tensor

  :param vocab_size: Number of characters in the input set vocabulary.
  :type vocab_size: int

  :param rand_func: Function to generate the
  :type rand_func: Callable

  :return: Output of the feed-forward network.
  :rtype: tf.Tensor
  """
  for i in range(0, const.FF_DEPTH):
    # For the first hidden layer, use the input layer
    if i == 0:
      input_width = input.shape[1]
      ff_in = input
    # Otherwise, use the previous layer
    else:
      input_width = const.FF_HIDDEN_WIDTH
      # noinspection PyUnboundLocalVariable
      ff_in = hidden_out

    bias_input = rand_func(tf.shape([const.FF_HIDDEN_WIDTH, 1]))
    hidden_layer = rand_func(tf.shape([input_width, const.FF_HIDDEN_WIDTH]))
    hidden_out = tf.add(tf.matmul(ff_in, hidden_layer), bias_input)

  # Construct the output layer
  bias_input = rand_func(tf.shape([vocab_size, 1]))
  out_layer = rand_func(tf.shape([const.FF_HIDDEN_WIDTH, vocab_size]))
  # noinspection PyUnboundLocalVariable
  return tf.add(tf.matmul(hidden_out, out_layer), bias_input)


def setup_feed_forward_and_softmax(input, vocab_size):
  """
  Feed-Forward and Softmax Builder

  Builds the feed-forward network and softmax layers.

  :param input: Item to be fed into the feed-forward network
  :type input: tf.Tensor

  :param vocab_size: Number of items in the vocabulary
  :type vocab_size: int

  :return: Output from the softmax
  :rtype: tf.Tensor
  """
  output_ff = _build_feed_forward(input, vocab_size, _get_tf_rand_uniform)

  softmax_out = tf.nn.softmax(output_ff)
  return softmax_out
