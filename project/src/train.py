import tensorflow as tf

import data_parser
import network
from const import Config


def run_training():
  input_str = data_parser.read_input()
  vocab_size = len(set(input_str))

  # get 1000 random examples
  sequences, targets = data_parser.get_examples(input_str, 1000)
  network_features = network.construct(vocab_size)

  init_op = tf.initialize_all_variables()
  with tf.Session() as sess:
    sess.run(init_op)


if __name__ == "__main__":
  Config.parse_args()
  run_training()
