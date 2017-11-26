import random

import tensorflow as tf
import logging
import data_parser
import network
from const import Config


def run_training():
  net_features = network.construct()

  X = net_features["X"]
  target = net_features["target"]
  seq_len = net_features["seq_len"]

  # Setup the training procedure
  cross_h = tf.nn.softmax_cross_entropy_with_logits(logits=net_features["output"],
                                                    labels=target)
  loss_op = tf.reduce_mean(cross_h)
  optimizer = tf.train.AdamOptimizer(learning_rate=Config.Train.learning_rate)
  train_op = optimizer.minimize(loss_op)

  sess = tf.Session()
  if Config.Train.restore:
    Config.import_model(sess)
  else:
    sess.run(tf.global_variables_initializer())

  for epoch in range(0, Config.Train.num_epochs):
    # Shuffle the batches for each epoch
    shuffled_list = list(range(Config.Train.size()))
    random.shuffle(shuffled_list)
    train_err = 0
    for batch in range(0, Config.Train.num_batch()):
      end_batch = min((batch + 1) * Config.Train.batch_size, Config.Train.size())
      start_batch = max(0, end_batch - Config.Train.batch_size)

      # Use the randomized batches
      train_x = list(map(lambda idx: Config.Train.x[idx], shuffled_list[start_batch:end_batch]))
      train_t = list(map(lambda idx: Config.Train.t[idx], shuffled_list[start_batch:end_batch]))
      seqlen = list(map(lambda idx: Config.Train.depth[idx], shuffled_list[start_batch:end_batch]))

      _, err = sess.run([train_op, loss_op], feed_dict={X: train_x, target: train_t,
                                                        seq_len: seqlen})
      train_err += err

    train_err /= Config.Train.num_batch()
    logging.info("Epoch %05d: Training Error: \t\t%0.3f" % (epoch, train_err))

    err = sess.run([loss_op], feed_dict={X: Config.Verify.x, target: Config.Verify.t})
    logging.info("Epoch %05d: Verification Error: \t%0.3f" % (epoch, err))

    if epoch % Config.Train.checkpoint_frequency == 0:
      Config.export_model(sess, epoch)

  sess.close()


if __name__ == "__main__":
  Config.parse_args()
  data_parser.build_training_and_verification_sets(dataset_size=200)

  run_training()
