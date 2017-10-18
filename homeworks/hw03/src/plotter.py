import matplotlib.pyplot as plt
from run_learner import build_lambdas, _build_random_results
from widgets import learning_alg_radio, k_slider, learning_rate_slider, regularizer_radio


def create_plots(training_errors, validation_errors, test_errors):
  x = build_lambdas()

  _verify_data_sizes(training_errors, validation_errors, test_errors)

  # Plot the training and validation errors
  plt.errorbar(x, training_errors[:, 0], training_errors[:, 1], label="Training")
  plt.errorbar(x, validation_errors[:, 0], validation_errors[:, 1], label="Validation")

  # Plot the test error (no error bars)
  plt.plot(x, test_errors[:, 0], label="Test")

  plot_title = "Effect of $\lambda$ on Learning Errors using %s" % learning_alg_radio.value
  plt.title(plot_title)

  # Label the plot information
  plt.rc('text', usetex=True)  # Enable Greek letters in MatPlotLib
  plt.xlabel("$\lambda$")
  plt.xscale('log')
  plt.ylabel("RMS Error")
  plt.yscale('log')
  plt.loglog()
  plt.legend(shadow=True, fontsize='x-large', loc="southeast")

  plt.show()
  filename = "error_%s_%s_k=%d_alpha=%f.pdf" % (learning_alg_radio.value, regularizer_radio.value,
                                                k_slider.value, learning_rate_slider.value)
  filename.replace(" ", "_")
  plt.savefig(".pdf")


def _verify_data_sizes(training_errors, validation_errors, test_errors):
  """
  Debug only code.  Verifies that the sizes of the training arrays are logical.
  :param training_errors:
  :param validation_errors:
  :param test_errors:
  :return:
  """
  # Verify the error matricies are correct
  lambdas_size = len(build_lambdas())
  for err_shape in [training_errors.shape, validation_errors.shape]:
    assert err_shape[0] == lambdas_size
    assert err_shape[1] == 2

  assert test_errors.shape[0] == lambdas_size
  assert test_errors.shape[1] == 1


if __name__ == "__main__":
  # Verify the plotter with random data
  training, validation, test = _build_random_results()
  create_plots(training, validation, test)
