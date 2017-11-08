import tensorflow as tf

def _init_weights_zero(shape):
  """
  Zero Weight initializer

  Initializes all weights/biases to value zero.

  :param shape: Shape of the weight vector
  :type shape: Tuple(int)

  :return: Zero vector for all the weights
  :rtype: tf.Tensor
  """
  weights = tf.zeros(shape)
  return tf.Variable(weights)


def init_weights_rand(shape):
  """
  Random Normal Weight initializer

  Initializes all weights/biases to a normal random value with standard
  deviation 0.1.

  :param shape: Shape of the weight vector
  :type shape: Tuple(int)

  :return: Zero vector for all the weights
  :rtype: tf.Tensor
  """
  weights = tf.random_normal(shape, stddev=0.1)
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
  hidden_layer = tf.nn.relu(tf.nn.sigmoid(tf.add(tf.matmul(X, weights['ff_hidden']), biases["ff_hidden"])))
  # Output fully connected layer with a neuron for each class
  out_layer = tf.matmul(hidden_layer, weights['ff_out']) + biases["ff_out"]
  return out_layer


def init(input_ff, n_classes):
  """
  Feed Forward Network Initializer

  Constructs the feed forward network components and returns the final logits.

  :param input_ff: Input tensor to the feed forward network.
  :return: Output of the feed-forward network.  It has not gone through the sigmoid
  function.
  :rtype: tf.Tensor
  """

  # Network Parameters
  n_input = input_ff.shape[1].value  # MNIST data input (img shape: 28*28)
  n_hidden = 128  # 1st layer number of neurons

  # Store layers weight & bias
  weights = {
    'ff_hidden': init_weights_rand([n_input, n_hidden]),
    'ff_out': init_weights_rand([n_hidden, n_classes])
  }
  biases = {
    'ff_hidden': init_weights_rand([n_hidden]),
    'ff_out': init_weights_rand([n_classes])
  }

  logits = _build_fully_connected_feed_forward(input_ff, weights, biases)
  return logits
