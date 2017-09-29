import numpy as np
import math

import matplotlib.pyplot as plt

enable_debug_graphing = False  # Enables some of the debug plotting
degree_of_interest = 5


class LambdaResults(object):

  def __init__(self):
    self._lambda_vals = []
    self._err_results = []

  def add_result(self, lambda_w, err_result):
    self._lambda_vals.append(lambda_w)
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
    x = self._lambda_vals  # x-axis data

    calc_mean = lambda results: [temp_res[degree].mean() for temp_res in results]
    calc_var = lambda results: [temp_res[degree].var() for temp_res in results]

    # Build the result data
    train_results = self._build_all_training()
    mean = []
    for res in train_results:
      data = res[degree]
      temp_mean = data.mean()
      mean.append(temp_mean)
    plt.errorbar(x, mean,
                 calc_var(train_results), label="training")

    validation_results = self._build_all_validation()
    plt.errorbar(x, calc_mean(validation_results),
                 calc_var(validation_results), label="validation")

    test_errs = [result.test[degree] for result in self._err_results]
    plt.plot(x, test_errs, label="test")

    # Define the graph information
    plt.xlabel("$\lambda$")
    plt.ylabel("RMS Error")
    y_min, y_max = plt.ylim()
    plt.ylim(ymin=max(y_min, 0), ymax=min(y_max, 5))  # Error is always positive
    plt.rc('text', usetex=True)  # Enable Greek letters in MatPlotLib
    plt.title("Effect of $\lambda$ on the Learning Errors using a %d-Degree Polynomial" % degree)
    plt.legend()
    # plt.show()
    plt.savefig("effect_lambda_for_degree=%02d_polynomial.pdf" % degree, bbox_inches='tight')
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
    y_min, _ = plt.ylim()
    plt.ylim(ymin=max(y_min,0))  # Error is always positive
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
  LAMBDA_STEP = .1  # ToDo: Lambda step size is too large
  LAMBDA_MIN = 0
  LAMBDA_MAX = 1
  lambda_range = np.arange(LAMBDA_MIN, LAMBDA_MAX, LAMBDA_STEP)

  # Create the results structure
  err_results = LambdaResults()

  # Test all the lambda.
  for lambda_w in lambda_range:
    lambda_err = _cross_validate_degree(params.min_degree, params.max_degree, params.k, lambda_w,
                                        all_train_inputs, all_train_t, all_test_inputs, all_test_t)
    err_results.add_result(lambda_w, lambda_err)
  global degree_of_interest
  for d in range(0, 20, 1):
    err_results.plot(d)


def _cross_validate_degree(min_degree, max_degree, k, lambda_w,
                           all_train_inputs, all_train_t,
                           all_test_inputs, all_test_t):
  """
  Runs k-Fold cross validation using the specified test and training data.

  :param min_degree: Minimum degree of the polynomial.  Must be greater than zero.
  :param max_degree:
  :param k: Number of folds in the cross validation.
  :type k: int
  :param lambda_w: Regularization constant
  :type lambda_w: float
  :param all_train_inputs:
  :param all_train_t:
  :param all_test_inputs:
  :param all_test_t:
  :return:
  :rtype: ErrorsStruct
  """
  global enable_debug_graphing

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
    enable_debug_graphing = False
    _, err_results.test[degree] = _determine_training_and_test_error(lambda_w, all_train_x, all_train_t,
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
  identity_matrix = np.identity(degree + 1, dtype=np.float64)

  # w* = ((X(X^T) - lambda * I)^-1) Xt
  x_transpose_product = np.matmul(train_x, train_x.transpose())
  w_star = np.linalg.inv(x_transpose_product + lambda_w * identity_matrix)
  w_star = np.matmul(np.matmul(w_star, train_x), train_t)

  # Debug Code for Looking at the Training Result
  global enable_debug_graphing, degree_of_interest
  if enable_debug_graphing and degree == degree_of_interest:
    x = test_x[1, :]
    plt.scatter(x, test_t, label="Test Target")
    plt.scatter(x, np.matmul(test_x.transpose(), w_star), label="Test Predicted")

    x = train_x[1, :]
    plt.scatter(x, train_t, label="Train Target")
    plt.scatter(x, np.matmul(train_x.transpose(), w_star), label="Train Predicted")

    plt.ylabel("X")
    plt.ylabel("Y")
    plt.legend()
    y_min, y_max = plt.ylim()

    plt.ylim(ymin=max(0,y_min), ymax=min(y_max, 30))
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

  >>> _calc_rms_error(np.matrix([2,3]).reshape(2,1), np.matrix([[2,4,6],[3,5,7]]).reshape(2,3), np.matrix([13,23,33]).reshape(3,1))
  0.0
  """
  n = x.shape[1]
  errs = np.matmul(x.transpose(), w_star)
  errs = errs - t
  e_w_star = 0.5 * errs.transpose() * errs
  return math.sqrt(2.0 * e_w_star[0] / n)


def _build_input_data_matix(degree, input_data, first_row=0, last_row=None):
  """
  Creates a (degree+1) by n matrix (where n = last_row-first_row) that stores the polynomial data
  used as "X" when solving for the ideal weight function.

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
  output = np.zeros([degree + 1, num_rows], dtype=np.float64)

  # Transfer to a data matrix
  output[0, :] = np.ones(num_rows, dtype=np.float64)  # Offset term
  x = input_data[first_row:last_row]
  for d in xrange(1, degree+1):
    output[d, :] = np.power(x, d).transpose()
  return output


if __name__ == "__main__":
  import doctest
  doctest.testmod()
