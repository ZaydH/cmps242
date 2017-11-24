#=====================================================================#
# network.py
# construct RNN for character level text prediction
#=====================================================================#

import tensorflow as tf
import data_parser
from decision_engine import setup_decision_engine
from feed_forward_and_softmax import setup_feed_forward_and_softmax
from const import Config

def construct_network(vocab_size):
    """
    Neural Network Constructor

    Builds all layers of the neural network.

    :param vocab_size: Number of elements in the vocabulary.
    :type vocab_size: int
    """

    # create data input placeholder
    X = tf.placeholder(tf.int32, shape=(None, None))

    # create target input placeholder
    Y = tf.placeholder(tf.int32, shape=(None))

    # create a random embedding matrix 
    embed_matrix = tf.Variable(
            tf.random_uniform(
                    [vocab_size, Config.EMBEDDING_SIZE], -1.0, 1.0
                )
        )

    # create the embedding lookup
    embedded = tf.nn.embedding_lookup(embed_matrix, X)

    # create RNN cell
    lstm_cell = tf.contrib.rnn.LSTMCell(Config.RNN_HIDDEN_SIZE, state_is_tuple = True)

    print(lstm_cell, embedded)

    # get rnn outputs
    rnn_output, rnn_state = tf.nn.dynamic_rnn(lstm_cell, embedded, dtype=tf.float32)

    print(rnn_output)

    # transpose rnn_output into a time major form
    rnn_output = tf.transpose(rnn_output, [1, 0, 2])
    # get the output of the last time-step
    rnn_final_output = tf.gather(rnn_output, Config.WINDOW_SIZE - 1)

    softmax_out = setup_feed_forward_and_softmax(rnn_final_output, vocab_size)

    decision_engine_out = setup_decision_engine(softmax_out)


    return {
            'X': X, 'Y': Y, 'RNN_OUTPUT': rnn_final_output
        }

def run_network(feature_dict, sequences, targets):
    """
    run a tensor flow session and try feeding the network stuff.
    just for testing right now
    """
    
    # start the session
    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)

# main
if __name__ == '__main__':

    input_str = data_parser.read_input()
    vocab_size = len(set(input_str))
    
    # get 1000 random examples
    sequences, targets = data_parser.get_examples(input_str, 1000)

    # 
    network_features = construct_network(vocab_size)

    # 
    run_network(network_features, sequences, targets)