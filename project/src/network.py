"""
 network.py

 Construct RNN for character level text prediction
"""

import tensorflow as tf
import data_parser
from feed_forward_and_softmax import setup_feed_forward_and_softmax
from basic_config import Config


def construct():
    """
    Trump Neural Network Constructor

    Builds all layers of the neural network.
    """

    # create data input placeholder
    input_x = tf.placeholder(tf.int32, shape=[Config.batch_size, None])

    # create target input placeholder
    target = tf.placeholder(tf.float32, shape=[Config.batch_size, Config.vocab_size()])

    # Create the embedding matrix
    embed_matrix = tf.get_variable("word_embeddings",
                                   [Config.vocab_size(), Config.RNN.hidden_size])
    embedded = tf.nn.embedding_lookup(embed_matrix, input_x)

    # create RNN cell
    cells = []
    for _ in Config.RNN.num_layers:
        cells.append(tf.nn.rnn_cell.BasicLSTMCell(Config.RNN.hidden_size, state_is_tuple=True))

    # get rnn outputs
    seq_len = tf.placeholder(tf.int32, shape=[Config.batch_size])
    rnn_output, rnn_state = tf.nn.dynamic_rnn(cell, embedded,
                                              sequence_length=seq_len,
                                              dtype=tf.float32)

    # transpose rnn_output into a time major form
    seq_end = tf.range(Config.batch_size) * tf.shape(rnn_output)[1] + (seq_len - 1)
    rnn_final_output = tf.gather(tf.reshape(rnn_output, [-1, Config.RNN_HIDDEN_SIZE]), seq_end)

    softmax_out = setup_feed_forward_and_softmax(rnn_final_output)

    final_output = softmax_out
    return {'X': input_x, 'target': target, 'RNN_OUTPUT': rnn_final_output,
            'seq_len': seq_len, 'output': final_output, 'embedding': embed_matrix}


def run():
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

    data_parser.build_training_and_verification_sets()
    network_features = construct()

    #
    run()
