import numpy as np
import matplotlib.pyplot as plt


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

  def _calc_training_mean(self):
    return self.training.mean(1)

  def _calc_training_var(self):
    return self.training.var(1)

  def _calc_validation_mean(self):
    return self.validation.mean(1)

  def _calc_validation_var(self):
    return self.validation.var(1)

  def plot(self):
    x = range(self._min_degree, self._max_degree + 1)
    plt.errorbar(x, self._calc_training_mean(),
                 self._calc_training_var(), label="training")
    plt.errorbar(x, self._calc_validation_mean(),
                 self._calc_validation_var(), 0.2, label="validation")
    plt.errorbar(x, self.test, 0.2, label="test")
    plt.show()


def run(params):
  """
  Calculates the training and test error for different number of degree polynomials

  :param params: Regression parameters
  """

  all_train_inputs = params.training_data[:, 0]
  all_train_t = params.training_data[:, 1]
  all_test_t = params.test_data[:, 1]

  size_training = all_train_inputs.size
  fold_size = size_training/params.k

  lambda_w = 0  # ToDo: Fix the value of lambda

  # Initialize the training error
  num_degrees = params.max_degree + 1 - params.min_degree
  err = ErrorsStruct(params.min_degree, num_degrees, params.k)

  # Test each degree
  for degree in xrange(params.min_degree, params.max_degree+1):
    degree_id = degree - params.min_degree

    # Perform the cross validation.
    for fold_id in xrange(0, params.k):
      fold_start = fold_id * fold_size
      fold_end = (fold_start + fold_size) if fold_id < params.k - 1 else size_training

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
      err.training[degree_id,fold_id] = fold_err[0]
      err.validation[degree_id, fold_id] = fold_err[0]

    # Determine the test error for the current degree
    all_train_x = _build_input_data_matix(degree, all_train_inputs)
    test_x = _build_input_data_matix(degree, params.test_data[:, 0])
    # No need for the full training error.
    _, err.test[degree] = _determine_training_and_test_error(lambda_w, all_train_x, all_train_t,
                                                             test_x, all_test_t)
  err.plot()
  x = 2


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

  # Calculate the errors and return it
  calc_error = lambda x, t: np.linalg.norm(x.transpose() * w_star - t, 2) + lambda_w * np.linalg.norm(w_star[1:degree+1])
  train_error = calc_error(train_x, train_t)
  test_error = calc_error(test_x, test_t)

  return train_error, test_error


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
