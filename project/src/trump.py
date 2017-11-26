"""

"""
import network
from const import Config
import tensorflow as tf
import logging


def generate_text():
  net_features = network.construct()

  X = net_features["X"]
  target = net_features["target"]
  seq_len = net_features["seq_len"]

  sess = tf.Session()
  Config.import_model(sess)

  generated_text = ""
  while len(generated_text) < Config.Generate.output_len:
    # Use the randomized batches
    # train_x = list(map(lambda idx: Config.Train.x[idx], shuffled_list[start_batch:end_batch]))
    # train_t = list(map(lambda idx: Config.Train.t[idx], shuffled_list[start_batch:end_batch]))
    # seqlen = list(map(lambda idx: Config.Train.depth[idx], shuffled_list[start_batch:end_batch]))
    train_x = Config.Train.x[start_batch:end_batch]
    train_t = Config.Train.t[start_batch:end_batch]
    seqlen = Config.Train.depth[start_batch:end_batch]
    _, err = sess.run(feed_dict={X: train_x, seq_len: seqlen})
    train_err += err

    # Delete off the front of the list if it has reached the specified sequence length
    if len(input_x[0]) == Config.sequence_length:
      del input_x[0][0]
    input_x[9].append(pred_char)

  sess.close()
  logging.info("Output Text:\t\t" + Config.Generate.seed + generated_text)


if __name__ == "__main__":
  Config.parse_args()
  Config.import_character_to_integer_map()
  Config.Generate.build_int2char()
  Config.Generate.build_seed_x()

  generate_text()
