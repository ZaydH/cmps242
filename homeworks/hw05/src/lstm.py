import tensorflow as tf
import input_parser
import const
import gensim
from hw05 import extract_train_and_test, _build_output_file
from feed_forward import init
import embedding_matrix
import numpy as np

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
  test_sequence_lengths = [len(l) for l in test_x] + [0 for i in range(len(train_x) - len(test_x))]

  # pad the examples with zeros
  max_len = max([len(x) for x in train_x])
  for i,x in enumerate(train_x):
    for _ in range(max_len - len(x)):
      train_x[i].append(0)

  # assert that the dimensions are now consistent
  assert len(set([len(x) for x in train_x])) == 1

  # pad the examples with zeros
  max_len = max([len(x) for x in test_x])
  for i,x in enumerate(test_x):
    for _ in range(max_len - len(x)):
      test_x[i].append(0)

  # assert that the dimensions are now consistent
  assert len(set([len(x) for x in test_x])) == 1

  test_x = np.array(test_x,dtype=np.int32)
  TEST_INSTANCES = len(test_x)
  # add bunch of all zero columns to test X
  test_x = np.vstack((
    test_x,
    np.zeros((len(train_x) - len(test_x),max_len))
  ))

  # create placeholder for RNN input
  X = tf.placeholder(tf.int32, shape=[const.BATCH_SIZE,None])

  # create a placeholder for the length of the sequences
  seqlen = tf.placeholder(tf.int32, shape=[const.BATCH_SIZE])

  # get the rnn inputs
  embeddings = tf.get_variable('embedding_matrix',[len(vocab), const.HIDDEN_SIZE])
  rnn_inputs = tf.nn.embedding_lookup(embeddings,X)
 
  # target placeholder
  targets = tf.placeholder(
    tf.float32,
    shape=(const.BATCH_SIZE,2)
  )

  # create cell
  cell = tf.nn.rnn_cell.BasicLSTMCell(
    const.HIDDEN_SIZE,
    state_is_tuple=True
  )

  # create an initial state
  # initial_state = cell.zero_state(tf.shape(rnn_inputs)[1], tf.float32)

  # construct a connected RNN
  rnn_outputs, rnn_state = tf.nn.dynamic_rnn(
    cell,
    rnn_inputs,
    # initial_state=initial_state,
    sequence_length=seqlen,
    dtype=tf.float32
  )

  # get the last rnn outputs
  idx = tf.range(const.BATCH_SIZE)*tf.shape(rnn_outputs)[1] + (seqlen - 1)
  final_rnn_outputs = tf.gather(tf.reshape(rnn_outputs, [-1, const.HIDDEN_SIZE]), idx)

  # make the rnn outputs the inputs to a FF network
  logits = init(final_rnn_outputs,2)

  # Softmax layer
  # with tf.variable_scope('sigmoid'):
  #     W = tf.get_variable('W', [const.HIDDEN_SIZE,1])
  #     b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0.0))
  
  # logits = tf.reshape(tf.matmul(final_rnn_outputs, W) + b,[const.BATCH_SIZE,-1])
  preds = tf.nn.sigmoid(logits)
  
  # correct = tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), tf.cast(targets,tf.int32))
  correct = tf.equal(tf.less_equal(preds,0.5),tf.equal(tf.cast(targets,tf.int32),0))
  accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
  
  # define loss function to be sigmoid
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=targets))

  # make a gradient descent optimizer
  train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)

  # create a session
  sess = tf.Session()
  # For some reason it is our job to do this:
  sess.run(tf.global_variables_initializer())
# `sess.graph` provides access to the graph used in a `tf.Session`.
  writer = tf.summary.FileWriter("/tmp/log/...", sess.graph)

  # # 
  # while 1:
  # # run
  #   print(session.run(
  #     [accuracy,loss,train_step],
  #     {
  #       X: train_x[:const.BATCH_SIZE],
  #       targets: train_y[:const.BATCH_SIZE],
  #       seqlen: sequence_lengths[:const.BATCH_SIZE]
  #     }
  #   ))
  
  # Perform the training
  for epoch in range(const.NUM_EPOCHS):
    # Run optimization op (backprop) and cost op (to get loss value)
    acc_err,_, c = sess.run(
      [accuracy,train_step, loss],
      {
        X: train_x[:const.BATCH_SIZE],
        targets:train_y[:const.BATCH_SIZE],
        seqlen: train_sequence_lengths[:const.BATCH_SIZE]
      }
    )
    print("Epoch: ", '%04d' % (epoch + 1), "cost={:.9f}".format(c))
    p_val = sess.run(preds, feed_dict={X: test_x, seqlen:test_sequence_lengths})
    p_val = p_val[:TEST_INSTANCES]
    # of = open("results_%04d.csv" % (epoch + 1),'w')
    # of.write('id,realDonaldTrump,HillaryClinton\n')
    # for i,j in enumerate(p_val):
    #   of.write('{},{:.8f},{:.8f}\n'.format(i,1-j[0],j[0]))
    # of.close()
    # classified = sess.run(accuracy, feed_dict={X: train_x, seqlen:train_sequence_lengths,targets:train_y})
    
    _build_output_file(p_val, output_file="results_%04d.csv" % (epoch + 1))

    # acc_err = np.mean(np.abs(classified - train_y))
    print("Training accuracy: %.2f%%" % (acc_err))

  print("Training Complete.")

  writer.close()
  sess.close()

  # get the tokenized sentences
  # sentences = [t.split() for t in train['tweet'] + test['tweet']]
  # Create the inputs
  
  # get the word embedding
  # embedding = gensim.models.Word2Vec(
  #   sentences,
  #   size=100,
  #   window=5,
  #   min_count=5,
  #   workers=4
  # )

  # Create the inputs
  # train_X, train_T, test_X, full_vocab = extract_train_and_test()
  # num_classes = 2

  # X = tf.placeholder("float", shape=[None,None,len(full_vocab)])
  
  # target = tf.placeholder("float", shape=[None, num_classes])

  # lstm_inputs = embedding_matrix.init(X, full_vocab)

  # create the RNN
  # create_lstm(lstm_inputs)