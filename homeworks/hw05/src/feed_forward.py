import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights_zero(shape):
  """
  Weight initializer

  :param shape: Shape of the weight vector
  :type shape: Tuple(int)

  :return: Zero vector for all the weights
  :rtype: tf.Tensor
  """
  weights = tf.zeros(shape)
  return tf.Variable(weights)


def forward_prop(X, w_hidden, w_out):
  """
  Forward-propagation.
  IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
  """
  hidden_layer = tf.nn.sigmoid(tf.matmul(X, w_hidden))  # The \sigma function
  yhat = tf.matmul(hidden_layer, w_out)  # The \varphi function
  return yhat


def init_feedforward(input_x):

  input_size = input_x.shape[1]         # Number of input nodes: Depending on if from LSTM or direct
  h_size = max(input_x.shape[1], 256)   # Number of hidden nodes - Make equaivalent to d
  a_out_size = 1                        # Single class Hillary/Trump

  # Symbols
  X = tf.placeholder("float", shape=[None, input_size])
  y = tf.placeholder("float", shape=[None, a_out_size])

  # Weight initializations
  w_1 = init_weights_zero((input_size, h_size))
  w_2 = init_weights_zero((h_size, y_size))

  # Forward propagation
  a_out = forward_prop(X, w_1, w_2)
  predict = tf.argmax(yhat, axis=1)

  # Backward propagation
  cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
  updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
