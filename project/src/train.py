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
    # shuffled_list = list(range(Config.Train.size()))
    # random.shuffle(shuffled_list)
    train_err = 0
    for batch in range(0, Config.Train.num_batch()):
      end_batch = min((batch + 1) * Config.batch_size, Config.Train.size())
      start_batch = max(0, end_batch - Config.batch_size)
      #
      # Use the randomized batches
      # train_x = list(map(lambda idx: Config.Train.x[idx], shuffled_list[start_batch:end_batch]))
      # train_t = list(map(lambda idx: Config.Train.t[idx], shuffled_list[start_batch:end_batch]))
      # seqlen = list(map(lambda idx: Config.Train.depth[idx], shuffled_list[start_batch:end_batch]))
      train_x = Config.Train.x[start_batch:end_batch]
      train_t = Config.Train.t[start_batch:end_batch]
      seqlen = Config.Train.depth[start_batch:end_batch]
      _, err = sess.run([train_op, loss_op], feed_dict={X: train_x, target: train_t,
                                                        seq_len: seqlen})
      train_err += err

    # ToDo It would be nice to add perplexity here.
    train_err /= Config.Train.num_batch()
    logging.info("Epoch %05d: Training Error: \t\t%0.3f" % (epoch, train_err))

    test_err = _calculate_verify_error(sess, loss_op, X, target, seq_len)
    logging.info("Epoch %05d: Verification Error: \t%0.3f" % (epoch, test_err))

    if epoch > 0 and epoch % Config.Train.checkpoint_frequency == 0:
      Config.export_model(sess, epoch)

  sess.close()


def _calculate_verify_error(sess, loss_op, X, target, seq_len):
  """
  Determines the verification error
  """
  verify_err = 0
  for batch in range(0, Config.Verify.num_batch()):
    end_batch = min((batch + 1) * Config.batch_size, Config.Verify.size())
    start_batch = max(0, end_batch - Config.batch_size)

    # Use the randomized batches
    verify_x = Config.Verify.x[start_batch:end_batch]
    verify_t = Config.Verify.t[start_batch:end_batch]
    seqlen = Config.Verify.depth[start_batch:end_batch]
    err = sess.run(loss_op, feed_dict={X: verify_x, target: verify_t, seq_len: seqlen})
    verify_err += err
  verify_err /= Config.Verify.num_batch()
  return verify_err


if __name__ == "__main__":
  Config.parse_args()
  data_parser.build_training_and_verification_sets(dataset_size=12500)

  run_training()
