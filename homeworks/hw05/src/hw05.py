import tensorflow as tf

import const
import input_parser
import numpy as np
import feed_forward


USE_BAG_OF_WORDS = True


def _build_output_file(p_trump, output_file="results.csv"):
  """
  Results File Creator

  Creates the results file for the Kaggle submission.

  :param p_trump: Probability that each tweet was written by @realDonaldTrump
  :type: np.array
  :param output_file: File to write the results file.
  :type output_file: str
  """
  with open(output_file, "w") as fout:
    fout.write("id,realDonaldTrump,HillaryClinton")
    for i in range(0, p_trump.shape[1]):
      assert(p_trump[i] >= 0 and p_trump[i] <= 1)
      fout.write("\n" + str(i) + ",")
      fout.write(p_trump[i] + "," + (1 - p_trump[i]))


def _flatten_one_hot(df):
  """
  One-Hot Converter

  Converts the one-hot vector to a bag of words representation.

  :param df: Test or training data frame.
  :type df: pd.DataFrame
  """
  # one_hot = enc.transform(train_data[const.COL_TWEET_TRANSFORM])
  df[const.COL_BAG_WORDS] = df[const.COL_ONE_HOT].apply(lambda x: np.sum(x, 0))


def extract_train_and_test():
  """
  Test and Training Data Extractor

  Helper function for extracting the test and training data.  It provides
  additional flexibility for if one-hot is desired or not.

  :return: Tuple(np.ndarray)
  """
  train, test, full_vocab = input_parser.parse()
  t_col = const.COL_TARGET
  if USE_BAG_OF_WORDS:
    _flatten_one_hot(train)
    _flatten_one_hot(test)
    num_train = train[t_col].size
    num_test = test[t_col].size
    x_col = const.COL_BAG_WORDS
    return train.as_matrix(columns=[x_col]), \
           train.as_matrix(columns=[t_col]), \
           test.as_matrix(columns=[x_col]), \
           test.as_matrix(columns=[t_col]), full_vocab
  else:
    x_col = const.COL_ONE_HOT




def init_tf():
  print("Using a fixed seed for Tensorflow's randomizer for ease of use.")
  random_seed = 0
  tf.set_random_seed(random_seed)

  init = tf.global_variables_initializer()
  return init


def run():
  """
  HW05 Master Function

  Runs the machine learning algorithm.
  """
  # Create the inputs
  train_X, train_T, test_X, test_T, full_vocab = extract_train_and_test()
  num_classes = 1  # Hillary v. Trump

  X = tf.placeholder("float", shape=[None, len(full_vocab)])
  target = tf.placeholder("float", shape=[None, num_classes])

  # Build the network
  input_ff = X
  logits = feed_forward.init(input_ff)
  predict = tf.sigmoid(logits)

  # Backward propagation
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=logits))  # Built-in Sigmoid
  updates = tf.train.GradientDescentOptimizer(const.LEARNING_RATE).minimize(cost)

  # Run SGD
  sess = tf.Session()
  init = init_tf()
  sess.run(init)

  # Perform the training
  for epoch in range(const.NUM_EPOCHS):
    # Train with each example
    for i in range(len(train_X)):
      if USE_BAG_OF_WORDS:
        sess.run(updates, feed_dict={X: train_X[i, :], target: train_T[i, :]})
      else:
        assert False  # Not yet supported

    train_accuracy = np.mean(np.argmax(train_T, axis=1) ==
                             sess.run(predict, feed_dict={X: train_X, target: train_T}))
    test_accuracy = np.mean(np.argmax(test_T, axis=1) ==
                            sess.run(predict, feed_dict={X: test_X, target: test_T}))

    print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
          % (epoch + 1, 100. * train_accuracy[0], 100. * test_accuracy[0]))


if __name__ == "__main__":
  run()