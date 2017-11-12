import tensorflow as tf
import input_parser
import const
import gensim
from hw05 import extract_train_and_test
import embedding_matrix
import numpy as np

# create recurrent neural network
def create_lstm(inputs):
  # create cell
  cell = tf.nn.rnn_cell.BasicLSTMCell(
    const.HIDDEN_SIZE,
    state_is_tuple=True
  )
  # single word input node
  # inputs = tf.placeholder(
  #   tf.float32,
  #   shape=(None,None,const.WORD_SIZE)
  # )
  # output node
  outputs = tf.placeholder(
    tf.float32,
    shape=(None,None,const.RNN_OUTPUT_SIZE)
  )
  # get the size of the batch, which is always (3,) i.e. empty 3d
  batch_size = tf.shape(inputs)[1]
  # create an initial state
  initial_state = cell.zero_state(batch_size, tf.float32)
  # construct a connected RNN
  net_ouput, net_state = tf.nn.dynamic_rnn(
    cell,
    inputs,
    initial_state=initial_state
  )

# main
if __name__ == '__main__':
  # Create the inputs
  train, test, vocab = input_parser.parse()

  # get the integer transformation of the train x
  train_x = [list(x) for x in train['int_transform'].values]

  # get the training targets
  train_y = np.array([[0 if y[0] else 1] for y in train['target']])

  # make sequence length vector 
  sequence_lengths = [len(l) for l in train_x]

  # pad the examples with zeros
  max_len = max([len(x) for x in train_x])
  for i,x in enumerate(train_x):
    for _ in range(max_len - len(x)):
      train_x[i].append(0)

  # assert that the dimensions are now consistent
  assert len(set([len(x) for x in train_x])) == 1

  train_x = np.array(train_x,dtype=np.int32)

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
    shape=(const.BATCH_SIZE,1)
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

  # Softmax layer
  with tf.variable_scope('sigmoid'):
      W = tf.get_variable('W', [const.HIDDEN_SIZE,1])
      b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0.0))
  
  logits = tf.reshape(tf.matmul(final_rnn_outputs, W) + b,[const.BATCH_SIZE,-1])
  preds = tf.nn.sigmoid(logits)
  # correct = tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), tf.cast(targets,tf.int32))
  correct = tf.equal(tf.less_equal(preds,0.5),tf.equal(tf.cast(targets,tf.int32),0))
  accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
  
  # define loss function to be sigmoid
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=targets))

  # make a gradient descent optimizer
  train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

  # create a session
  session = tf.Session()
  # For some reason it is our job to do this:
  session.run(tf.global_variables_initializer())

  # 
  while 1:
  # run
    print(session.run(
      [accuracy,loss,train_step],
      {
        X: train_x[:const.BATCH_SIZE],
        targets: train_y[:const.BATCH_SIZE],
        seqlen: sequence_lengths[:const.BATCH_SIZE]
      }
    )) 
  
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