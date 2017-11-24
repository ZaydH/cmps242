import tensorflow as tf


def run_session():
  init_op = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init_op)


if __name__ == "__main__":
