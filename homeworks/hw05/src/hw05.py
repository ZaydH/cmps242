import tensorflow as tf

import const
import input_parser
import numpy as np
import feed_forward
import embedding_matrix


USE_BAG_OF_WORDS = True


def _build_output_file(p_val, output_file="results.csv"):
  """
  Results File Creator

  Creates the results file for the Kaggle submission.

  :param p_val: Probability that each tweet was written by Donald Trump and Hillary
  :type: np.ndarray
  :param output_file: File to write the results file.
  :type output_file: str
  """
  with open(output_file, "w") as fout:
    fout.write("id,realDonaldTrump,HillaryClinton")
    for i, prob in enumerate(p_val):
      fout.write("\n" + str(i) + ",")
      if const.LBL_DONALD_TRUMP == [1, 0]:
        fout.write(str(prob[0]))
        fout.write("," + str(prob[1]))
      else:
        fout.write(str(1 - prob[1]))
        fout.write("," + str(prob[0]))

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
    x_col = const.COL_BAG_WORDS
    return np.matrix(train[x_col].tolist()), \
           np.matrix(train[t_col].tolist()), \
           np.matrix(test[x_col].tolist()), \
           full_vocab
  else:
    x_col = const.COL_ONE_HOT
    return train[x_col], train[t_col], test[x_col], full_vocab


def init_tf():
  init = tf.global_variables_initializer()
  return init


def run():
  """
  HW05 Master Function

  Runs the machine learning algorithm.
  """
  print("Using a fixed seed for Tensorflow's randomizer for ease of use.")
  random_seed = 0
  tf.set_random_seed(random_seed)

  # Create the inputs
  train_X, train_T, test_X, full_vocab = extract_train_and_test()
  num_classes = 2

  X = tf.placeholder("float", shape=[None, len(full_vocab)])
  input_ff = X
  target = tf.placeholder("float", shape=[None, num_classes])

  # Build the network
  input_ff = embedding_matrix.init(X, full_vocab)
  logits = feed_forward.init(input_ff, num_classes)
  predict = tf.sigmoid(logits)
  accuracy = tf.round(predict)

  # Define loss and optimizer
  loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=target))
  # optimizer = tf.train.AdamOptimizer(learning_rate=const.LEARNING_RATE)
  # train_op = optimizer.minimize(loss_op)
  global_step = tf.Variable(0, trainable=False)
  learning_rate = tf.train.exponential_decay(const.LEARNING_RATE, global_step,
                                             const.EPOCHS_PER_DECAY, const.DECAY_RATE, staircase=True)
  train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_op, global_step=global_step)

  # Run Gradient Descent
  sess = tf.Session()
  init = init_tf()
  sess.run(init)

  # `sess.graph` provides access to the graph used in a `tf.Session`.
  writer = tf.summary.FileWriter("/tmp/log/...", sess.graph)

  # Perform the training
  for epoch in range(const.NUM_EPOCHS):
    # Run optimization op (backprop) and cost op (to get loss value)
    _, c = sess.run([train_op, loss_op], feed_dict={X: train_X, target: train_T})
    print("Epoch: ", '%04d' % (epoch + 1), "cost={:.9f}".format(c))
    p_val = sess.run(predict, feed_dict={X: test_X})
    _build_output_file(p_val, output_file="results_%04d.csv" % (epoch + 1))
    classified = sess.run(accuracy, feed_dict={X: train_X})
    acc_err = np.mean(np.abs(classified - train_T))
    print("Training accuracy: %.2f%%" % (100. - 100. * acc_err))

  print("Training Complete.")

  writer.close()
  sess.close()


if __name__ == "__main__":
  run()
