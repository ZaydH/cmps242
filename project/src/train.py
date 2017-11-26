import tensorflow as tf

import data_parser
import network
from const import Config


def run_training():
  input_str = data_parser.read_input()

  # get 1000 random examples
  data_parser.create_examples(input_str, 1000)
  network.construct()

  init_op = tf.initialize_all_variables()
  with tf.Session() as sess:
    sess.run(init_op)


if __name__ == "__main__":
  Config.parse_args()
  data_parser.build_training_set()

  run_training()
