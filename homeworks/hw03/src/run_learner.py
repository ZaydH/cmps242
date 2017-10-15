import widgets
import random
import const
import input_parser
import numpy as np


def run_hw03(train_data, test_data):
  """
  HW03 Learner

  Performs the learning for homework 3.

  :param train_data: Matrix containing training feature and target values
  :type train_data: np.matrix

  :param test_data: Matrix containing test feature and target values
  :type test_data: np.matrix

  :return:
  :rtype: List[np.matrix]
  """
  # Extract the information from the widgets
  k = widgets.k_slider.value
  # Strategy pattern for which learner to run
  if widgets.learning_alg_radio.value == const.GD_ALG:
    learner_func = run_gradient_descent_learner
  elif widgets.learning_alg_radio.value == const.EG_ALG:
    learner_func = run_eg_learner
  else:
    raise ValueError("Invalid learning algorithm")
  num_train = train_data.shape[0]
  eta = widgets.learning_rate_slider.value
  lambdas = build_lambdas()

  loss_function = regularized_error # TODO Update support for multiple loss functions.

  # Build the results structures
  num_lambdas = len(lambdas)
  train_err = np.zeros([num_lambdas, 2])
  valid_err = np.zeros([num_lambdas, 2])
  test_err = np.zeros([num_lambdas, 1])

  # Build the indices for each of the folds.
  validation_sets = create_cross_validation_fold_sets(num_train, k)

  for idx, lambda_val in enumerate(lambdas):
    # Get cross validation results
    results = perform_cross_validation(train_data, validation_sets,
                                       learner_func, loss_function, eta, lambda_val)
    train_err[idx, :] = np.matrix(results[0:2])
    valid_err[idx, :] = np.matrix(results[2:])

    # Train on the full training set then verify against the test data.
    # Extract the training and test data
    train_x, train_t, test_x, test_t = _extract_train_and_test_data(train_data, test_data)
    # Get the test error
    w_star = learner_func(train_x, train_t, eta, lambda_val)
    test_err[idx] = calculate_rms_error(w_star, test_x, test_t)

  return train_err, valid_err, test_err


def perform_cross_validation(train_data, validation_sets,
                             learner_func, loss_function, eta, lambda_val):
  """
  Execute Cross-Validation

  :param train_data: Pandas DataFrame containing all TRAINING data with labels and features.
  :type train_data: pd.DataFrame

  :param validation_sets: Indices of the rows to be used for validation in each fold.
  :type validation_sets: List[List[int]]

  :param learner_func: Function that performs the learning
  :type learner_func: callable

  :param loss_function: Function used to calculate the new weights
  :type loss_function: callable

  :param eta: Learning rate
  :type eta: float

  :param lambda_val: Regularization value to be used
  :type lambda_val: float

  :return: Error mean and variable
  :rtype: List[float]
  """

  # Run cross validation
  k = len(validation_sets)
  training_sets = create_training_set_indices(validation_sets)

  # Initialize the storage for the weights
  train_err = np.zeros(k)
  validation_err = np.zeros(k)

  # Perform the K folds
  for fold_cnt in xrange(0, k):
    # Extract the training and test data
    train_x, train_t, valid_x, valid_t = _extract_train_and_test_data(train_data, train_data,
                                                                      training_sets[fold_cnt],
                                                                      validation_sets[fold_cnt])
    # Learn the function
    w_star = learner_func(train_x, train_t, loss_function, eta, lambda_val)
    train_err[fold_cnt] = calculate_rms_error(w_star, train_x, train_t)
    validation_err[fold_cnt] = calculate_rms_error(w_star, valid_x, valid_t)

  return np.mean(train_err), np.var(train_err), np.mean(validation_err), np.var(validation_err)


def calculate_rms_error(w_star, x_tensor, t_vec):
  """
  RMS Calculator

  Calculates the RMS error for the x_tensor and the target vector.

  :param w_star: Learned weight vector
  :type w_star: np.matrix

  :param x_tensor: Input feature tensor.
  :type x_tensor: np.matrix

  :param t_vec: Array of target values
  :type t_vec: np.matrix

  :return: RMS error
  :rtype: float
  """
  y = np.matmul(x_tensor, w_star)
  err = np.power(y - t_vec, 2)

  num_samples = x_tensor.shape[0]
  return np.sum(err) / num_samples


def run_gradient_descent_learner(train_x, train_t, loss_function, eta, lambda_val,
                                 num_epochs=25):
  """

  :param train_x: X-tensor to be learned.
  :type train_x: np.matrix

  :param train_t: Target value for the learner.
  :type train_t: np.matrix

  :param loss_function: Function used for calculating the loss
  :type loss_function: callable

  :param eta: Learning rate
  :type eta: float

  :param lambda_val: Lambda regularization value
  :type lambda_val: float

  :param num_epochs: Number of training epochs
  :type num_epochs: int

  :return: Final learned weight vector
  :rtype: np.matrix
  """
  n = train_x.shape[1]
  w = initialize_weights(n)
  for i in range(0, num_epochs):
    w -= eta * (i ** const.ALPHA) * loss_function(train_x, train_t, lambda_val)
  return w


def regularized_error(train_x, train_t, lambda_val):
  """
  Regularized Error Calculator

  :param train_x: Training X tensor
  :type train_x: np.matrix

  :param train_t: Training target value
  :type train_t: np.matrix

  :param lambda_val: Regularizer value
  :type lambda_val: float

  :return: Associated weight vector
  :rtype: np.matrix
  """
  identity_matrix = np.identity(train_x.shape[1], dtype=np.float64)
  identity_matrix[0,0] = 0  # Do not regularize the bias.

  # w* = ((X(X^T) - lambda * I)^-1) Xt
  x_transpose_product = np.matmul(train_x.transpose(), train_x)
  return np.linalg.solve(x_transpose_product + lambda_val * identity_matrix, np.matmul(train_x.transpose(), train_t))


def run_eg_learner(train_x, train_t, loss_function, lambda_val,
                   num_epochs=25):
  # TODO: Implement the EG learner
  pass


def build_lambdas():
  """
  Lambdas Builder

  Builds the values of lambda to test based off the widget slide.

  :return: Values of Lambda to test
  :rtype: List[float]
  """
  return [0] + [2 ** x for x in range(widgets.lambdas_range_slider.value[0],
                                      widgets.lambdas_range_slider.value[1]+1)]


def _build_random_results():
  """
  Debug Results Generator

  Builds debug results for testing while the program is being debugged.

  :return: Training, validation, and test errors in a list respectively.
  :rtype: List[np.matrix]
  """
  lambdas = build_lambdas()
  training = np.random.rand(len(lambdas), 2)
  training[:, 0] = training[:, 0] + 1
  validation = np.random.rand(len(lambdas), 2)
  validation[:, 0] = validation[:, 0] + 1

  test = np.random.rand(len(lambdas), 1)

  return training, validation, test


def create_cross_validation_fold_sets(n, k):
  """
  Cross-Validation Fold Builder

  This function is used to create the cross validation folds.
  All the folds are of equal or near equal size.

  :param n: Number of training samples.
  :type n: int
  :param k: Number of folds.
  :type k: int
  :return: List of the row indices for the k folds.
  :rtype: List[List[int]]
  """
  indices = range(0, n)
  random.shuffle(indices)

  splits = [[] for _ in xrange(0, k)]
  for i in xrange(0, n):
    splits[i % k].append(indices[i])
  # For ease of use, sort the validation set.
  for i in range(0, k):
    splits[i] = sorted(splits[i])
  return splits


def create_training_set_indices(validation_sets):
  """
  Cross-Validation Training Set Builder

  This function is used to build the training sets.
  It should be run after the function "create_cross_validation_fold_sets".

  :param validation_sets:
  :type validation_sets: List[List[int]]
  :return:
  :rtype: List[List[int]]
  """
  n = reduce(lambda x, y: x + len(y), validation_sets, 0)
  num_folds = len(validation_sets)
  splits = [[] for _ in xrange(0, num_folds)]
  # Iterate through the k folds
  for fold_cnt in xrange(0, num_folds):
    v_cnt = 0
    # Any samples not in the validation set should go into the
    # training set.
    for t_cnt in xrange(0, n):
      if v_cnt < len(validation_sets[fold_cnt]) and t_cnt == validation_sets[fold_cnt][v_cnt]:
        v_cnt += 1
        continue
      splits[fold_cnt].append(t_cnt)

    # Debug code - just checking I did this right
    all_examples = set()
    all_examples.update(validation_sets[fold_cnt])
    all_examples.update(splits[fold_cnt])
    assert len(all_examples) == n
  return splits


def initialize_weights(n):
  """
  Weight Vector Initializer

  The weight vector selected may differ depending on the specfic learning
  algorithm selected.

  :param n: Number of dimensions in the weight vector
  :type n: int
  :return: Initial weight vector.
  :rtype: np.matrix
  """
  return np.zeros([n, 1])


def _extract_train_and_test_data(train_df, test_df, train_row_indexes=None, test_row_indexes=None):
  """
  Test and Training Extractor

  This is used to extract the train and test/validation data.  It builds them into
  Numpy arrays.

  :param train_df: Pandas DataFrame containing the training target and feature data
  :type train_df: pd.DataFrame
  :param test_df: Pandas DataFrame containing the test target and feature data
  :type test_df: pd.DataFrame
  :param train_row_indexes: List of row indices to select for the training data
  :type train_row_indexes: List[int]
  :param test_row_indexes: List of row indices to select for the test data
  :type test_row_indexes: List[int]

  :return: Numpy matrices for trainX, trainT, testX, testT
  :rtype: List[np.matrix]
  """
  train_x, train_t = _build_x_and_target(train_df, train_row_indexes)

  test_x, test_t = _build_x_and_target(test_df, test_row_indexes)

  return train_x, train_t, test_x, test_t


def _build_x_and_target(data_mat, row_indexes=None):
  """
  Pandas to NumPy Converter

  :param data_mat: Matrix of the data.  First column is the target data. The rest is feature data/
  :type data_mat: np.matrix

  :param row_indexes: List of rows to selects.  Can be ignored if all the rows are desired.
  :type row_indexes: List[int]

  :return: Final learned weight vector
  :rtype: np.matrix
  """
  target_values = data_mat[:, 0]
  x_values = data_mat[:, 1:]
  # If appropriate, get only a subset of the rows
  if row_indexes is not None:
    rows_np = np.array(row_indexes)
    x_values = x_values[rows_np, :]
    target_values = target_values[rows_np, :]

  # Prepend an offset term as appropriate
  ones_vector = np.ones([x_values.shape[0], 1])
  x_values = np.hstack([ones_vector, x_values])

  return x_values, target_values


if __name__ == "__main__":
  train_examples, test_examples = input_parser.parse()
  run_hw03(train_examples, test_examples)
