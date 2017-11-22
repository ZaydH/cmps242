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

    # 

# main
if __name__ == '__main__':

    input_str = data_parser.read_input()
    vocab_size = len(set(input_str))
    
    # get 1000 random examples
    sequences, targets = data_parser.get_examples(input_str, 1000)

    # 
    construct_network(vocab_size)
