import tensorflow as tf

def _init_weights_zero(shape):
  """
  Weight initializer

  Initializes all weights/biases to value zero.

  :param shape: Shape of the weight vector
  :type shape: Tuple(int)

  :return: Zero vector for all the weights
  :rtype: tf.Tensor
  """
  weights = tf.zeros(shape)
  return tf.Variable(weights)

def _build_fully_connected_feed_forward(X, weights, biases):
  """
  Constructs the feed-forward network.

  :param X: Input to the feed-forward network
  :type X: tf.Placeholder
  :param weights: Weights between the different network layers
  :type weights: dict
  :param biases: Offsets/biases into each of the neurons
  :type biases: dict

  :return: Output tensor
  :rtype: tf.Tensor
  """
  # Hidden fully connected layer with 256 neurons
  hidden_layer = tf.add(tf.matmul(X, weights['w_ff_hidden']), biases['b_ff_hidden'])
  # Output fully connected layer with a neuron for each class
  out_layer = tf.add(tf.matmul(hidden_layer, weights['w_ff_out']), biases['b_ff_out'])
  return out_layer


def init(input_ff):
  """
  Feed Forward Network Initializer

  Constructs the feed forward network components and returns the final logits.

  :param input_ff: Input tensor to the feed forward network.
  :return: Output of the feed-forward network.  It has not gone through the sigmoid
  function.
  :rtype: tf.Tensor
  """

  # Network Parameters
  n_input = input_ff.shape[1]  # MNIST data input (img shape: 28*28)
  n_hidden = max(256, input_ff.shape[1])  # 1st layer number of neurons
  n_classes = 1  # Either Hillary or Trump

  # Store layers weight & bias
  weights = {
    'w_ff_hidden': _init_weights_zero([n_input, n_hidden]),
    'w_ff_out': _init_weights_zero([n_hidden, n_classes])
  }
  biases = {
    'b_ff_hidden': _init_weights_zero([n_hidden, 1]),
    'b_ff_out': _init_weights_zero([n_classes, 1])
  }

  logits = _build_fully_connected_feed_forward(input_ff, weights, biases)
  return logits
