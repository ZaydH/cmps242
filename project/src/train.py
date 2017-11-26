import tensorflow as tf

import data_parser
import network
from const import Config


def run_training():
  saver = tf.train.Saver()

  if not Config.Train.restart:
    Config.import_model()
  else:
    init_op = tf.initialize_all_variables()

  with tf.Session() as sess:
    sess.run(init_op)


if __name__ == "__main__":
  Config.parse_args()
  data_parser.build_training_and_verification_sets(dataset_size=1000)
  network.construct()


  run_training()
