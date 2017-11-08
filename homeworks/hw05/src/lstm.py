import tensorflow as tf
import input_parser
import const
import gensim

# create recurrent neural network
def create_lstm():
  # create cell
  cell = tf.nn.rnn_cell.BasicLSTMCell(
    const.HIDDEN_SIZE,
    state_is_tuple=True
  )
  # single word input node
  inputs = tf.placeholder(
    tf.float32,
    shape=(None,None,const.WORD_SIZE)
  )
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
  # get the tokenized sentences
  sentences = [t.split() for t in train['tweet'] + test['tweet']]
  # get the word embedding
  embedding = gensim.models.Word2Vec(
    sentences,
    size=100,
    window=5,
    min_count=5,
    workers=4
  )
  # create the RNN
  create_lstm()