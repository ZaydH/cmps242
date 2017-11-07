import tensorflow as tf
import input_parser
import numpy as np


def _build_output_file(p_trump, output_file="results.csv"):
  """
  Results File Creator

  Creates the results file for the Kaggle submission.

  :param p_trump: Probability that each tweet was written by @realDonaldTrump
  :type: np.array
  :param output_file: File to write the results file.
  :type output_file: str
  """
  with open(output_file, "w") as fout:
    fout.write("id,realDonaldTrump,HillaryClinton")
    for i in range(0, p_trump.shape[1]):
      assert(p_trump[i] >= 0 and p_trump[i] <= 1)
      fout.write("\n" + str(i) + ",")
      fout.write(p_trump[i] + "," + (1 - p_trump[i]))

def run():

  # Create the inputs
  train, test = input_parser.parse()


  # Run SGD for the classifier
  sess = tf.Session()
  init = tf.global_variables_initializer()
  sess.run(init)

  for epoch in range(100):
    # Train with each example
    for i in range(len(train_X)):
        sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

    train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                             sess.run(predict, feed_dict={X: train_X, y: train_y}))
    test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
                            sess.run(predict, feed_dict={X: test_X, y: test_y}))

    print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
          % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))



if __name__ == "__main__":
  run()