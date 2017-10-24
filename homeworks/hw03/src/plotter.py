import matplotlib.pyplot as plt
import const
import os
import run_learner
import widgets


def create_plots(training_errors, validation_errors, test_errors):
  x = run_learner.build_lambdas()
  # Make sure the errors and lambdas have matching sizes
  _verify_data_sizes(training_errors, validation_errors, test_errors)

  # Plot the training and validation errors
  plt.errorbar(x, training_errors[:, 0], training_errors[:, 1], label="Training")
  plt.errorbar(x, validation_errors[:, 0], validation_errors[:, 1], label="Validation")

  # Plot the test error (no error bars)
  plt.plot(x, test_errors[:, 0], label="Test")

  plot_title = "Effect of $\lambda$ on Learning Errors using %s" % widgets.learning_alg_radio.value
  plt.title(plot_title)

  # Label the plot information
  # plt.rc('text', usetex=True)  # Enable Greek letters in MatPlotLib
  plt.xlabel("$\lambda$")
  if widgets.error_type_radio.value == const.ERROR_RMS:
    plt.ylabel("RMS Error")
  elif widgets.error_type_radio.value == const.ERROR_ACCURACY:
    plt.ylabel("1 - Accuracy")
  else:
    raise ValueError("Unknown error type")
  plt.loglog()
  plt.legend(shadow=True, fontsize='x-large', loc='best')

  # # Calculate the axes so the log scale is easy to read
  # y_max = 1.11 * np.max([training_errors[:, 0] + training_errors[:, 1],
  #                       validation_errors[:, 0] + validation_errors[:, 1],
  #                       test_errors[:, 0]])
  # y_min = 0.89 * np.min([training_errors[:, 0] - training_errors[:, 1],
  #                       validation_errors[:, 0] - validation_errors[:, 1],
  #                       test_errors[:, 0]])
  # cur_y_min, cur_y_max = plt.ylim()
  # plt.ylim([min(cur_y_min, y_min), max(cur_y_max, y_max)])
  plt.tight_layout()  # Ensure no title/label is cutoff

  # Save the plot to a directory
  filename = run_learner.build_results_filename() + ".pdf"
  try:
    os.makedirs(const.IMG_DIR)
  except OSError:
    pass
  plt.savefig(const.IMG_DIR + os.sep + filename)

  # Output the graph to the screen
  plt.show()


def plot_eg_learning_rate(eg_errs):

  # Plot the training and validation errors
  labels = ["Training", "Validation", "Test"]
  for idx, label in enumerate(labels):
    plt.plot(eg_errs[0, :], eg_errs[idx + 1, :], label=label)

  plot_title = "Effect of Learning Rate $\eta_0$ on EG Learning Errors using %s" % const.ALG_EG
  plt.title(plot_title)

  # Label the plot information
  # plt.rc('text', usetex=True)  # Enable Greek letters in MatPlotLib
  plt.xlabel("Learning Rate ($\eta_0$)")
  if widgets.error_type_radio.value == const.ERROR_RMS:
    plt.ylabel("RMS Error")
  elif widgets.error_type_radio.value == const.ERROR_ACCURACY:
    plt.ylabel("1 - Accuracy")
  else:
    raise ValueError("Unknown error type")
  plt.loglog()
  plt.legend(shadow=True, fontsize='x-large', loc='best')
  plt.tight_layout()  # Ensure no title/label is cutoff

  # Save the plot to a directory
  filename = run_learner.build_results_filename() + ".pdf"
  try:
    os.makedirs(const.IMG_DIR)
  except OSError:
    pass
  plt.savefig(const.IMG_DIR + os.sep + filename)

  # Output the graph to the screen
  plt.show()


def _verify_data_sizes(training_errors, validation_errors, test_errors):
  """
  Debug only code.  Verifies that the sizes of the training arrays are logical.
  :param training_errors:
  :param validation_errors:
  :param test_errors:

  :return:
  """
  # Verify the error matricies are correct
  lambdas_size = len(run_learner.build_lambdas())
  for err_shape in [training_errors.shape, validation_errors.shape]:
    assert err_shape[0] == lambdas_size
    assert err_shape[1] == 2

  assert test_errors.shape[0] == lambdas_size
  assert test_errors.shape[1] == 1


if __name__ == "__main__":
  # Verify the plotter with random data
  # noinspection PyProtectedMember
  training, validation, test = run_learner._build_random_results()
  create_plots(training, validation, test)
