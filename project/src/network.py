#=====================================================================#
# network.py
# construct RNN for character level text prediction
#=====================================================================#

import tensorflow as tf
import const
import data_parser


def construct_network(vocab_size):
    """
    make neural net
    """

    # create data input placeholder
    X = tf.placeholder(tf.int32, shape=(None, None))

    # create target input placeholder
    Y = tf.placeholder(tf.int32, shape=(None))

    # create a random embedding matrix 
    embed_matrix = tf.Variable(
            tf.random_uniform(
                    [vocab_size, const.EMBEDDING_SIZE], -1.0, 1.0
                )
        )

    # create the embedding lookup
    embedded = tf.nn.embedding_lookup(embed_matrix, X)

    # create RNN cell
    lstm_cell = tf.contrib.rnn.LSTMCell(const.RNN_HIDDEN_SIZE, state_is_tuple = True)

    print(lstm_cell, embedded)

    # get rnn outputs
    rnn_output, rnn_state = tf.nn.dynamic_rnn(lstm_cell, embedded, dtype=tf.float32)

    print(rnn_output)

    # transpose rnn_output into a time major form
    rnn_output = tf.transpose(rnn_output, [1, 0, 2])
    
    # get the output of the last time-step
    rnn_final_output = tf.gather(
            rnn_output, const.WINDOW_SIZE - 1
        )

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