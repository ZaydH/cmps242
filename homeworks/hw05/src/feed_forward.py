import tensorflow as tf

import const


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
  assert const.NUM_HIDDEN_LAYERS >= 1  # Need at least one hidden layer
  # Hidden fully connected layer with 256 neurons
  hidden_layers = []
  hidden_layers.append(tf.nn.relu(tf.nn.sigmoid(tf.add(tf.matmul(X, weights['ff_hidden1']), biases["ff_hidden1"]))))

  # Create the hidden layers
  for i in range(2, const.NUM_HIDDEN_LAYERS+1):
    layer = tf.nn.relu(tf.nn.sigmoid(tf.add(tf.matmul(hidden_layers[i-2],
                                                      weights['ff_hidden' + str(i)]),
                                                      biases["ff_hidden" + str(i)])))
    hidden_layers.append(layer)

  # Output fully connected layer with a neuron for each class
  out_layer = tf.matmul(hidden_layers[const.NUM_HIDDEN_LAYERS - 1], weights['ff_out']) + biases["ff_out"]
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
  n_hidden = const.NUM_NEURON_HIDDEN_LAYER  # 1st layer number of neurons

  # Store layers weight & bias
  weights = {
    'ff_hidden1': init_weights_rand([n_input, n_hidden]),
    'ff_out': init_weights_rand([n_hidden, n_classes])
  }
  biases = {
    'ff_hidden1': init_weights_rand([n_hidden]),
    'ff_out': init_weights_rand([n_classes])
  }
  for i in range(2, const.NUM_HIDDEN_LAYERS+1):
    weights['ff_hidden' + str(i)] = init_weights_rand([n_hidden, n_hidden])
    biases['ff_hidden' + str(i)] = init_weights_rand([n_hidden])

  logits = _build_fully_connected_feed_forward(input_ff, weights, biases)
  return logits
