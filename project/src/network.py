#=====================================================================#
# network.py
# construct RNN for character level text prediction
#=====================================================================#

import tensorflow as tf
import data_parser
from decision_engine import setup_decision_engine
from feed_forward_and_softmax import setup_feed_forward_and_softmax
from const import Config


def construct():
    """
    Trump Neural Network Constructor

    Builds all layers of the neural network.
    """

    # create data input placeholder
    input_x = tf.placeholder(tf.int32, shape=(None, None))

    # create target input placeholder
    target = tf.placeholder(tf.int32, shape=(None))

    # create a random embedding matrix 
    embed_matrix = tf.Variable(
            tf.random_uniform(
                    [Config.vocab_size(), Config.EMBEDDING_SIZE], -1.0, 1.0
                )
        )

    # create the embedding lookup
    embedded = tf.nn.embedding_lookup(embed_matrix, input_x)

    # create RNN cell
    lstm_cell = tf.contrib.rnn.LSTMCell(Config.RNN_HIDDEN_SIZE, state_is_tuple=True)

    # get rnn outputs
    rnn_output, rnn_state = tf.nn.dynamic_rnn(lstm_cell, embedded, dtype=tf.float32)

    # transpose rnn_output into a time major form
    rnn_output = tf.transpose(rnn_output, [1, 0, 2])
    # get the output of the last time-step
    rnn_final_output = tf.gather(rnn_output, Config.WINDOW_SIZE - 1)

    softmax_out = setup_feed_forward_and_softmax(rnn_final_output)

    # In training mode, the decision engine is removed and compare directly
    # to the softmax output.
    if not Config.is_train():
        decision_engine_out = setup_decision_engine(softmax_out)
        final_output = decision_engine_out
    else:
        final_output = softmax_out
    return {'X': input_x, 'target': target, 'RNN_OUTPUT': rnn_final_output,
            'output': final_output}


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
