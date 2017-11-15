import tensorflow as tf
import input_parser
import const
import gensim
from hw05 import extract_train_and_test, _build_output_file
from feed_forward import init
import embedding_matrix
import numpy as np

NUM_BATCHES = 10

# main
if __name__ == '__main__':
  # Create the inputs
  train, test, vocab = input_parser.parse()

  # get the integer transformation of the train x
  train_x = [list(x) for x in train['int_transform'].values]

  # get the integer transformation of the train x
  test_x = [list(x) for x in test['int_transform'].values]

  # get the training targets
  # train_y = np.array([[0 if y[0] else 1] for y in train['target']])
  train_y = np.array([np.array(y) for y in train['target']])

  # make the batch size the whole thing
  const.BATCH_SIZE = len(train_y)

  # make sequence length vector s
  train_sequence_lengths = [len(l) for l in train_x]
  test_sequence_lengths = [len(l) for l in test_x]

  # pad the examples with zeros
  max_seq_len = max(max(train_sequence_lengths), max(test_sequence_lengths))
  for seq_lens, x_vals in [(train_sequence_lengths, train_x),
                           (test_sequence_lengths, test_x)]:
    for x_val in x_vals:
      while len(x_val) < max_seq_len:
        x_val.append(0)
    # assert that the dimensions are now consistent
    assert len({len(x) for x in x_vals}) == 1

  # create placeholder for RNN input
  X = tf.placeholder(tf.int32, shape=[None, max_seq_len])

  # create a placeholder for the length of the sequences
  seqlen = tf.placeholder(tf.int32, shape=[None])

  # get the rnn inputs
  embeddings = tf.get_variable('embedding_matrix', [len(vocab), const.HIDDEN_SIZE])
  rnn_inputs = tf.nn.embedding_lookup(embeddings, X)
 
  # target placeholder
  targets = tf.placeholder(tf.float32, shape=[None, 2])

  # create cell
  lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
    const.HIDDEN_SIZE,
    state_is_tuple=True
  )

  # create an initial state
  batch_size = tf.shape(X)[0]
  initial_state = lstm_cell.zero_state(batch_size, tf.float32)

  # construct a connected RNN
  rnn_outputs, rnn_state = tf.nn.dynamic_rnn(
    lstm_cell,
    rnn_inputs,
    initial_state=initial_state,
    sequence_length=seqlen,
    dtype=tf.float32
  )

  # get the last rnn outputs
  idx = tf.range(batch_size)*tf.shape(rnn_outputs)[1] + (seqlen - 1)
  final_rnn_outputs = tf.gather(tf.reshape(rnn_outputs, [-1, const.HIDDEN_SIZE]), idx)

  # make the rnn outputs the inputs to a FF network
  logits = init(final_rnn_outputs, 2)

  preds = tf.nn.sigmoid(logits)
  accuracy = tf.round(preds)

  # define loss function to be sigmoid
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets))

  # make a gradient descent optimizer
  global_step = tf.Variable(0, trainable=False)
  const.LEARNING_RATE = .01
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
    # Run optimization op (backprop) and cost op (to get loss value)
    for batch_num in range(0, NUM_BATCHES):
      start_batch = batch_num * TRAIN_BATCH_SIZE
      end_batch = (batch_num + 1) * TRAIN_BATCH_SIZE if batch_num != NUM_BATCHES - 1 else const.BATCH_SIZE
      acc_err, _, c = sess.run([accuracy, train_step, loss],
                               feed_dict={X: train_x[start_batch:end_batch],
                                          targets: train_y[start_batch:end_batch],
                                          seqlen: train_sequence_lengths[start_batch:end_batch]})

    classified, c = sess.run([accuracy, loss], feed_dict={X: train_x, targets: train_y, seqlen: train_sequence_lengths})
    print("Epoch: ", '%04d' % (epoch + 1), "cost={:.9f}".format(c))

    # Store the probability results
    p_val = sess.run(preds, feed_dict={X: test_x, seqlen: test_sequence_lengths})
    _build_output_file(p_val, output_file="results_%04d.csv" % (epoch + 1))

    # Print the training classification accuracy
    acc_err = np.mean(np.abs(classified - train_y))
    print("Training accuracy: %.2f%%" % (100. - 100. * acc_err))
  print("Training Complete.")

  writer.close()
  sess.close()
