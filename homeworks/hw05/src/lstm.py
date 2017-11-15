import tensorflow as tf
import input_parser
import const
from hw05 import build_output_file
from feed_forward import init
import numpy as np


# main
if __name__ == '__main__':
  # # Create the inputs
  train, test, vocab, dummy_word = input_parser.parse()

  # get the integer transformation of the train x
  train_x = [list(x) for x in train[const.COL_WORD2VEC].values]

  # get the integer transformation of the train x
  test_x = [list(x) for x in test[const.COL_WORD2VEC].values]

  # get the training targets
  train_y = np.array([np.array(y) for y in train['target']])

  # make sequence length vector s
  train_sequence_lengths = [len(l) for l in train_x]
  test_sequence_lengths = [len(l) for l in test_x]

  # pad the examples with dummy data
  max_seq_len = max(max(train_sequence_lengths), max(test_sequence_lengths))
  for x_vals in [train_x, test_x]:
    for x_val in x_vals:
      while len(x_val) < max_seq_len:
        x_val.append(dummy_word)
  train_w2v = [np.vstack(x_val) for x_val in train_x]
  test_w2v = [np.vstack(x_val) for x_val in test_x]

  num_test_inputs = len(test_w2v)
  max_input_size = max(len(train_w2v), num_test_inputs)
  # Pad the test set
  while len(test_w2v) < max_input_size:
    test_w2v.append(test_w2v[0])
    test_sequence_lengths.append(0)

  # create placeholder for RNN input
  X = tf.placeholder(tf.float32, shape=[max_input_size, max_seq_len, const.HIDDEN_SIZE])

  # create a placeholder for the length of the sequences
  seqlen = tf.placeholder(tf.int32, shape=[max_input_size])

  # target placeholder
  targets = tf.placeholder(tf.float32, shape=[max_input_size, 2])

  # create cell
  lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(const.HIDDEN_SIZE, state_is_tuple=True)

  # construct a connected RNN
  rnn_outputs, rnn_state = tf.nn.dynamic_rnn(lstm_cell, X,
                                             sequence_length=[max_seq_len] * max_input_size,
                                             dtype=tf.float32)

  # get the last rnn outputs
  idx = tf.range(max_input_size)*tf.shape(rnn_outputs)[1] + (seqlen - 1)
  final_rnn_outputs = tf.gather(tf.reshape(rnn_outputs, [-1, const.HIDDEN_SIZE]), idx)

  # make the rnn outputs the inputs to a FF network
  logits = init(final_rnn_outputs, 2)
  preds = tf.nn.sigmoid(logits)
  accuracy = tf.round(preds)

  # define loss function to be sigmoid
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets))

  # make a gradient descent optimizer
  global_step = tf.Variable(0, trainable=False)
  const.LEARNING_RATE = 1.
  learning_rate = tf.train.exponential_decay(const.LEARNING_RATE, global_step,
                                             const.EPOCHS_PER_DECAY, const.DECAY_RATE, staircase=True)
  # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
  train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)

  # create a session
  sess = tf.Session()
  # For some reason it is our job to do this:
  sess.run(tf.global_variables_initializer())
# `sess.graph` provides access to the graph used in a `tf.Session`.
  writer = tf.summary.FileWriter("/tmp/log/...", sess.graph)

  # Perform the training
  TRAIN_BATCH_SIZE = const.BATCH_SIZE // NUM_BATCHES
  for epoch in range(const.NUM_EPOCHS):
    acc_err, _, c = sess.run([accuracy, train_step, loss],
                             feed_dict={X: train_w2v,
                                        targets: train_y,
                                        seqlen: train_sequence_lengths})
    classified, c = sess.run([accuracy, loss], feed_dict={X: train_x, targets: train_y, seqlen: train_sequence_lengths})
    print("Epoch: ", '%04d' % (epoch + 1), "cost={:.9f}".format(c))

    # Store the probability results
    p_val = sess.run(preds, feed_dict={X: test_w2v, seqlen: test_sequence_lengths})
    build_output_file(p_val[:num_test_inputs], output_file="results_%04d.csv" % (epoch + 1))

    # Print the training classification accuracy
    acc_err = np.mean(np.abs(classified - train_y))
    print("Training accuracy: %.2f%%" % (100. - 100. * acc_err))
  print("Training Complete.")

  writer.close()
  sess.close()
