import numpy as np
import math

import matplotlib.pyplot as plt

enable_debug_graphing = False  # Enables some of the debug plotting


class LambdaResults(object):

  def __init__(self):
    self._k_values = []
    self._err_results = []

  def add_result(self, k, err_result):
    self._k_values.append(k)
    self._err_results.append(err_result)

  def _build_all_training(self):
    return [result.training for result in self._err_results]

  def _build_all_validation(self):
    return [result.validation for result in self._err_results]

  def plot(self, degree):
    """
    Generates a plot of the effect of lambda on the accuracy.

    :param degree: Polynomial degree to plot.
    :type degree: int
    """
    x = self._k_values

    calc_mean = lambda results, d: [res.mean(1)[d] for res in results]
    calc_var = lambda results, d: [res.var(1)[d] for res in results]
    # Build the result data
    train_results = self._build_all_training()
    plt.errorbar(x, calc_mean(train_results, degree),
                 calc_var(train_results, degree), label="training")

    validation_results = self._build_all_validation()
    plt.errorbar(x, calc_mean(validation_results, degree),
                 calc_var(validation_results, degree), label="validation")

    test_errs = [result.test[degree] for result in self._err_results]
    plt.plot(x, test_errs, label="test")

    # Define the graph information
    plt.xlabel("Polynomial Degree")
    plt.ylabel("RMS Error")
    plt.ylim(ymin=0)  # Error is always positive
    plt.rc('text', usetex=True)  # Enable Greek letters in MatPlotLib
    plt.title("Effect of $\lambda$ on the Training, Validation, and Test Errors")
    plt.show()
    plt.close()


class ErrorsStruct(object):

  def __init__(self, min_degree, num_degrees, k):
    """
    Creates a Errors structure.

    :param min_degree: Minimum polynomial order tested
    :type min_degree: int
    :param num_degrees: Number of different polynomial orders tested
    :type num_degrees: int
    :param k: Number of folds
    :type k: int
    """
    self._min_degree = min_degree
    self._max_degree = min_degree + num_degrees - 1
    self.training = np.zeros([num_degrees, k])
    self.validation = np.zeros([num_degrees, k])
    self.test = np.zeros(num_degrees)

  def calc_training_mean(self):
    return self.training.mean(1)

  def calc_training_var(self):
    return self.training.var(1)

  def calc_validation_mean(self):
    return self.validation.mean(1)

  def calc_validation_var(self):
    return self.validation.var(1)

  def plot(self):
    """
    Plot the relationship between polynomial order and error
    """

    x = range(self._min_degree, self._max_degree + 1)
    plt.errorbar(x, self.calc_training_mean(),
                 self.calc_training_var(), label="training")
    plt.errorbar(x, self.calc_validation_mean(),
                 self.calc_validation_var(), label="validation")
    plt.plot(x, self.test, label="test")

    # Setup the graph itself
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Regularized Error")
    plt.ylim(ymin=0)  # Error is always positive
    plt.legend()
    plt.show()


# noinspection PyPep8Naming
def run(params):
  """
  Calculates the training and test error for different number of degree polynomials

  :param params: Regression parameters
  """

  # Separate the independent and dependent variables
  all_train_inputs = params.training_data[:, 0]
  all_train_t = params.training_data[:, 1]
  all_test_inputs = params.test_data[:, 0]
  all_test_t = params.test_data[:, 1]

  # Define the lambda settings
  LAMBDA_STEP = 5  # ToDo: Lambda step size is too large
  LAMBDA_MIN = 0
  LAMBDA_MAX = 13
  lambda_range = np.arange(LAMBDA_MIN, LAMBDA_MAX, LAMBDA_STEP)

  # Create the results structure
  err_results = LambdaResults()

  # Test all the lambda.
  for lambda_w in lambda_range:
    lambda_err = _cross_validate_degree(params.min_degree, params.max_degree, params.k, lambda_w,
                                        all_train_inputs, all_train_t, all_test_inputs, all_test_t)
    err_results.add_result(lambda_w, lambda_err)
  DEGREE_TO_PLOT = 19
  err_results.plot(DEGREE_TO_PLOT)


def _cross_validate_degree(min_degree, max_degree, k, lambda_w,
                           all_train_inputs, all_train_t,
                           all_test_inputs, all_test_t):

  size_training = all_train_inputs.size
  fold_size = size_training / k

  # Initialize the training error
  num_degrees = max_degree + 1 - min_degree
  err_results = ErrorsStruct(min_degree, num_degrees, k)

  # Test each degree
  for degree in xrange(min_degree, max_degree+1):
    degree_id = degree - min_degree

    # Perform the cross validation.
    for fold_id in xrange(0, k):
      fold_start = fold_id * fold_size
      fold_end = (fold_start + fold_size) if fold_id < k - 1 else size_training

      # Build x and t
      concat_test_data = np.concatenate([all_train_inputs[0:fold_start],
                                         all_train_inputs[fold_end:size_training]])
      train_x = _build_input_data_matix(degree, concat_test_data)
      train_t = np.concatenate([all_train_t[0:fold_start],
                                all_train_t[fold_end:size_training]])

      validation_x = _build_input_data_matix(degree, all_train_inputs, fold_start, fold_end)
      validation_t = all_train_t[fold_start:fold_end]

      # Determine the validation error
      fold_err = _determine_training_and_test_error(lambda_w, train_x, train_t,
                                                    validation_x, validation_t)
      err_results.training[degree_id, fold_id] = fold_err[0]
      err_results.validation[degree_id, fold_id] = fold_err[1]

    # Determine the test error for the current degree
    all_train_x = _build_input_data_matix(degree, all_train_inputs)
    test_x = _build_input_data_matix(degree, all_test_inputs)
    # No need for the full training error.
    # ToDo: Determine why test error is the least
    global enable_debug_graphing
    enable_debug_graphing = True
    _, err_results.test[degree] = _determine_training_and_test_error(lambda_w, train_x, train_t,
                                                                     test_x, all_test_t)
    enable_debug_graphing = False

  if enable_debug_graphing:
    err_results.plot()
  return err_results


def _determine_training_and_test_error(lambda_w, train_x, train_t, test_x, test_t):
  """
  Trains a model based off the specified data and returns the training and test error.

  :param lambda_w: Lambda model scalar.
  :type lambda_w: float
  :param train_x: Training X tensor with additional rows for the polynomial value
  :type train_x: numpy.ndarray
  :param train_t: Training target values
  :type train_t: numpy.ndarray
  :param test_x: Test X tensor with additional rows for the polynomial
  :type test_x: numpy.ndarray
  :param test_t: Test set target values
  :type test_t: numpy.ndarray

  :return: A tuple containing the training and test error respectively.
  :rtype: Tuple[float]
  """

  degree = train_x.shape[0] - 1
  identity_matrix = np.identity(degree + 1, dtype=np.float32)

  # w* = ((X(X^T) - lambda * I)^-1) Xt
  w_star = np.linalg.pinv(np.matmul(train_x, train_x.transpose()) + lambda_w * identity_matrix)
  w_star = np.matmul(w_star, train_x) * train_t

  # Debug Code for Looking at the Training Result
  global enable_debug_graphing
  if enable_debug_graphing and degree > 0:
    x = test_x[1, :]
    plt.scatter(x, test_t, label="Test Target")
    plt.scatter(x, np.matmul(test_x.transpose(), w_star), label="Test Predicted")

    x = train_x[1, :]
    plt.scatter(x, train_t, label="Train Target")
    plt.scatter(x, np.matmul(train_x.transpose(), w_star), label="Train Predicted")

    plt.ylabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
    plt.close()

  # Calculate the errors and return it
  train_error = _calc_rms_error(w_star, train_x, train_t)
  test_error = _calc_rms_error(w_star, test_x, test_t)

  return train_error, test_error


def _calc_rms_error(w_star, x, t):
  """
  Simple Helper Function for RMS Error
  Calculates the E_RMS as = sqrt(2E(w)

  :param w_star: Weight vector
  :type w_star: numpy.ndarray
  :param x: Input data tensor
  :type x: numpy.ndarray
  :param t: Input data vector
  :type t: numpy.ndarray
  :return: Error
  :rtype: float
  """
  n = x.shape[1]
  e_w_star = np.linalg.norm(x.transpose() * w_star - t, 2)
  return math.sqrt(2 * e_w_star / n)


def _build_input_data_matix(degree, input_data, first_row=0, last_row=None):
  """

  :param degree: Polynomial degree
  :type degree: int

  :param input_data:
  :type input_data:

  :param first_row: Index of the first row from the array input_data to take (inclusive)
  :type first_row: int

  :param last_row: Upper end (exclusive) of the input_data to transfer to a matrix
  :type last_row: int

  :return: Input polynomial matrix of size num_rows (degree+1) by (last_row - first_row)
  :rype: numpy.matrix
  """
  if last_row is None:
    last_row = len(input_data)

  num_rows = last_row - first_row
  output = np.zeros([degree + 1, num_rows], dtype=np.float32)

  # Transfer to a data matrix
  output[0, :] = np.ones(num_rows, dtype=np.float32)  # Offset term
  x = input_data[first_row:last_row]
  for d in xrange(1, degree+1):
    output[d, :] = np.power(x, d).transpose()
  return output
