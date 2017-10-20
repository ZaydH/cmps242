# -*- coding: utf-8 -*-
import widgets
import random
import const
import input_parser
import numpy as np
import math
import sys


def run_hw03(train_data, test_data):
  """
  HW03 Learner

  Performs the learning for homework 3.

  :param train_data: Matrix containing training feature and target values
  :type train_data: np.ndarray

  :param test_data: Matrix containing test feature and target values
  :type test_data: np.ndarray

  :return: Training, validation, and test error respectively.
  :rtype: Tuple[np.ndarray]
  """
  np.seterr(over='ignore')  # Mask the numpy overflow warning.
  print("Starting Learner.")
  # Extract the information from the widgets
  k = widgets.k_slider.value
  # Strategy pattern for which learner to run
  add_plus_minus_data = False
  if widgets.learning_alg_radio.value == const.ALG_GD:
    learner_func = run_gd_learner
  elif widgets.learning_alg_radio.value == const.ALG_SGD:
    learner_func = run_sgd_learner
  elif widgets.learning_alg_radio.value == const.ALG_EG:
    learner_func = run_eg_learner
    add_plus_minus_data = True
  else:
    raise ValueError("Invalid learning algorithm")
  # Strategy Pattern for the Regularizer
  if widgets.regularizer_radio.value == const.REGULARIZER_L1_NORM:
    regularizer_func = l1_norm_regularizer
  elif widgets.regularizer_radio.value == const.REGULARIZER_L2_NORM:
    regularizer_func = l2_norm_regularizer
  else:
    raise ValueError("Invalid regularizer")

  # ToDo Decide what to do about regularization of the EG+/- Learner
  if widgets.learning_alg_radio.value == const.ALG_EG:
    regularizer_func = zero_regularizer
  # Strategy Pattern for the Regularizer
  if widgets.error_type_radio.value == const.ERROR_RMS:
    calculate_validation_err_func = calculate_rms_error
  elif widgets.error_type_radio.value == const.ERROR_ACCURACY:
    calculate_validation_err_func = calculate_accuracy
  else:
    raise ValueError("Invalid validation error function")
  num_train = train_data.shape[0]
  eta = widgets.learning_rate_slider.value
  lambdas = build_lambdas()

  # Build the results structures
  num_lambdas = len(lambdas)
  train_err = np.zeros([num_lambdas, 2])
  valid_err = np.zeros([num_lambdas, 2])
  test_err = np.zeros([num_lambdas, 1])

  # Build the indices for each of the folds.
  validation_sets = create_cross_validation_fold_sets(num_train, k)
  # Train on the full training set then verify against the test data.
  # Extract the training and test data
  train_x, train_t, test_x, test_t = _extract_train_and_test_data(train_data, test_data,
                                                                  convert_to_plus_minus=add_plus_minus_data)
  for idx, lambda_val in enumerate(lambdas):
    print_percent_done(current_fold=0, current_lambda=lambda_val)
    # Get cross validation results
    results = perform_cross_validation(train_data, validation_sets,
                                       learner_func, regularizer_func, calculate_validation_err_func,
                                       eta, lambda_val, add_plus_minus_data,)
    train_err[idx, :] = np.matrix(results[0:2])
    valid_err[idx, :] = np.matrix(results[2:])

    # Get the test error
    w_star = learner_func(train_x, train_t, regularizer_func, eta, lambda_val)
    test_err[idx] = calculate_validation_err_func(w_star, test_x, test_t)

  # Record progress complete.
  print_percent_done(current_fold=widgets.k_slider.value + 1,
                     current_lambda=lambdas[-1])
  print("Learner complete.")
  return train_err, valid_err, test_err


def perform_cross_validation(train_data, validation_sets,
                             learner_func, regularizer_func, calculate_validation_err_func,
                             eta, lambda_val, convert_to_plus_minus):
  """
  Execute Cross-Validation

  :param train_data: Pandas DataFrame containing all TRAINING data with labels and features.
  :type train_data: pd.DataFrame

  :param validation_sets: Indices of the rows to be used for validation in each fold.
  :type validation_sets: List[List[int]]

  :param learner_func: Function that performs the learning
  :type learner_func: callable

  :param regularizer_func: Function to regularize the error
  :type regularizer_func: callable

  :param calculate_validation_err_func: Calculates the validation error
  :type calculate_validation_err_func: callable

  :param eta: Learning rate
  :type eta: float

  :param lambda_val: Regularization value to be used
  :type lambda_val: float

  :param convert_to_plus_minus: Selects whether to add a negative version of the input to the extracted data
  :type convert_to_plus_minus: bool

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
                                                                      validation_sets[fold_cnt],
                                                                      convert_to_plus_minus)
    # Learn the function
    w_star = learner_func(train_x, train_t, regularizer_func, eta, lambda_val)
    train_err[fold_cnt] = calculate_validation_err_func(w_star, train_x, train_t)
    validation_err[fold_cnt] = calculate_validation_err_func(w_star, valid_x, valid_t)

    # Calculate the percent done and report it.
    print_percent_done(fold_cnt + 1, lambda_val)
  return np.mean(train_err), np.var(train_err), np.mean(validation_err), np.var(validation_err)


def print_percent_done(current_fold, current_lambda):
  """
  Prints the percent done based off the current fold and the current lambda under test.

  :param current_fold: Fold number
  :type current_fold: int
  :param current_lambda: Current lambda under test
  :type current_lambda: float
  """
  k = widgets.k_slider.value
  all_lambdas = build_lambdas()
  numb_runs = (k + 1) * len(all_lambdas)
  lambda_cnt = all_lambdas.index(current_lambda)
  training_cnt = current_fold + lambda_cnt * (k + 1)
  print_progress(iteration=training_cnt, total=numb_runs, prefix="Progress: ")


def calculate_rms_error(w_star, x_tensor, t_vec):
  """
  RMS Calculator

  Calculates the RMS error for the x_tensor and the target vector.

  :param w_star: Learned weight vector
  :type w_star: np.ndarray

  :param x_tensor: Input feature tensor.
  :type x_tensor: np.ndarray

  :param t_vec: Array of target values
  :type t_vec: np.ndarray

  :return: RMS error
  :rtype: float
  """
  y_hat = sigmoid_vec(np.matmul(x_tensor, w_star))
  y_diff = y_hat - t_vec
  err = np.power(y_diff, 2)
  sum_err = np.sum(err)
  rms = sum_err / x_tensor.shape[0]
  return math.sqrt(rms)


def calculate_accuracy(w_star, x_tensor, t_vec):
  """
  RMS Calculator

  Calculates the RMS error for the x_tensor and the target vector.

  :param w_star: Learned weight vector
  :type w_star: np.ndarray

  :param x_tensor: Input feature tensor.
  :type x_tensor: np.ndarray

  :param t_vec: Array of target values
  :type t_vec: np.ndarray

  :return: RMS error
  :rtype: float
  """
  y_hat = sigmoid_vec(np.matmul(x_tensor, w_star))
  y_hat_round = np.round(y_hat, 0)
  y_diff = np.abs(np.subtract(y_hat_round, t_vec))
  return np.sum(y_diff) / x_tensor.shape[0]


def run_gd_learner(train_x, train_t, regularizer_func, eta, lambda_val):
  """
  Standard Gradient Descent Learner

  Performs the gradient descent algorithm.

  :param train_x: X-tensor to be learned.
  :type train_x: np.ndarray

  :param train_t: Target value for the learner.
  :type train_t: np.ndarray

  :param regularizer_func: Function used to run different regularizers
  :type regularizer_func: callable

  :param eta: Learning rate
  :type eta: float

  :param lambda_val: Lambda regularization value
  :type lambda_val: float

  :return: Final learned weight vector
  :rtype: np.ndarray
  """
  w = initialize_weights_gd(train_x.shape[1])
  for t in range(1, widgets.epoch_slider.value + 1):  # Starting from zero is not possible because of aging term t ^ \alpha
    w_prev = np.copy(w)
    w_star = gd_regularizer_error(w_prev, train_x, train_t, lambda_val, regularizer_func)
    w_change = eta * (t ** (-const.ALPHA)) * w_star
    w -= w_change

    # Allow premature exit.
    max_change = np.max(np.abs(np.subtract(w, w_prev)))
    if max_change < 10 ** -3:
      break
  return w


def gd_regularizer_error(w, train_x, train_t, lambda_val, regularizer_func):
  """
  Gradient Descent Regularized Error Calculator

  Calculates the derivative of the gradient descent loss function using the
  form:

  err = (\hat{y} - y)^T X^T + \lambda w_t

  :param w: Weight vector
  :type w: np.ndarray

  :param train_x: Training X tensor
  :type train_x: np.ndarray

  :param train_t: Training target value
  :type train_t: np.ndarray

  :param lambda_val: Regularizer value
  :type lambda_val: float

  :param regularizer_func:
  :type regularizer_func: callable

  :return: Associated weight vector
  :rtype: np.ndarray
  """
  y_hat = sigmoid_vec(np.matmul(train_x, w))
  err = np.subtract(y_hat, train_t)
  prod = np.matmul(err.transpose(), train_x).transpose()

  # Calculate the regularizer
  regularizer_err = regularizer_func(lambda_val, w)

  return np.add(prod, regularizer_err)


def l1_norm_regularizer(lambda_val, w):
  """
  L1 Norm Regularizer

  Calculates the L1 regularizer.  It removes the bias term.

  :param lambda_val: Regularizer term
  :type lambda_val: float
  :param w: Weight vector
  :type w: np.ndarray

  :return: Regularized err term
  :rtype: np.ndarray
  """
  reg_err = np.multiply(np.ones(w.shape), lambda_val)
  reg_err[0, 0] = 0  # Exclude the bias
  return reg_err


def l2_norm_regularizer(lambda_val, w):
  """
  L2 Norm Regularizer

  Calculates the L2 regularizer.  It removes the bias term.

  :param lambda_val: Regularizer term
  :type lambda_val: float
  :param w: Weight vector
  :type w: np.ndarray

  :return: Regularized err term
  :rtype: np.ndarray
  """
  reg_err = np.multiply(lambda_val, w)
  reg_err[0, 0] = 0  # Exclude the bias
  return reg_err


def sigmoid_vec(z):
  """
  Vectorized Sigmoid Function Calculator

  Calculates the sigmoid function on a NumPy array.

  :param z: Input vector for the sigmoid function
  :type z: np.ndarray

  :return: \sigma(z)
  :rtype: np.ndarray
  """
  # noinspection SpellCheckingInspection
  denom = np.add(1, np.exp(-1 * z))
  return np.divide(1, denom)


def run_eg_learner(train_x, train_t, regularizer_func, eta, lambda_val):
  """
  Exponentiated Gradient +/- Learner

  Performs the exponentiated gradient algorithm.

  :param train_x: X-tensor to be learned.
  :type train_x: np.ndarray

  :param train_t: Target value for the learner.
  :type train_t: np.ndarray

  :param regularizer_func: Function used to run different regularizers
  :type regularizer_func: callable

  :param eta: Learning rate
  :type eta: float

  :param lambda_val: Lambda regularization value
  :type lambda_val: float

  :return: Final learned weight vector
  :rtype: np.ndarray
  """
  w = initialize_weights_eg(train_x.shape[1])
  for t in range(1, widgets.epoch_slider.value + 1):  # Starting from zero is not possible because of aging term t ^ \alpha
    w_prev = np.copy(w)
    exp_weight = _eg_regularized_error(w, train_x, train_t, eta, lambda_val, regularizer_func)
    w = np.multiply(w_prev, exp_weight)

    # Normalize all the weights
    w_sum = np.sum(w)
    w = np.divide(w, w_sum)
    _verify_eg_w_length(w)

    # Allow premature exit.
    max_change = np.max(np.abs(np.subtract(w, w_prev)))
    if max_change < 10 ** -3:
      break
  return w


def _eg_regularized_error(w_t, train_x, train_t, eta, lambda_val, regularizer_func):
  """
  Exponentiated Regularizer Error

  Calculates the regularized error for the exponentiated gradient
  algorithm.

  :param w_t: Current weight vector
  :type w_t: np.ndarray
  :param train_x: X tensor
  :type train_x: np.ndarray
  :param train_t: Target training value
  :type train_t: np.ndarray
  :param eta: Learning rate
  :type eta: float
  :param lambda_val: Regularizer value
  :type lambda_val: float
  :param regularizer_func:
  :type regularizer_func: callable
  :return:
  :rtype: np.ndarray
  """
  n = train_x.shape[0]  # Number of training examples
  y_hat = sigmoid_vec(np.matmul(train_x, w_t))

  err = np.subtract(y_hat, train_t)
  regularizer_err = regularizer_func(lambda_val, w_t)
  prod = np.divide(np.matmul(train_x.transpose(), err) + regularizer_err, n)
  exp_weight = np.multiply(-1 * eta, prod)
  return np.exp(exp_weight)


# noinspection PyUnusedLocal
def zero_regularizer(lambda_val, w_t):
  """
  Empty regularizer that has no effect

  :param lambda_val: Regularizer scalar
  :type lambda_val: float
  :param w_t: Weight vector for the current epoch
  :type w_t: np.ndarray
  :return: Regularizer correct
  :rtype: np.ndarray
  """
  return np.zeros(w_t.shape)


def run_sgd_learner(train_x, train_t, regularizer_func, eta, lambda_val):
  """
  Stochastic Gradient +/- Learner

  Performs the stochastic gradient descent algorithm.

  :param train_x: X-tensor to be learned.
  :type train_x: np.ndarray

  :param train_t: Target value for the learner.
  :type train_t: np.ndarray

  :param regularizer_func: Function used to run different regularizers
  :type regularizer_func: callable

  :param eta: Learning rate
  :type eta: float

  :param lambda_val: Lambda regularization value
  :type lambda_val: float

  :return: Final learned weight vector
  :rtype: np.ndarray
  """
  w = initialize_weights_gd(train_x.shape[1])

  scrambled_range = range(0, train_x.shape[0])
  # Starting from zero is not possible because of aging term t ^ \alpha
  for t in range(1, widgets.epoch_slider.value + 1):
    random.shuffle(scrambled_range)
    for sample_row in scrambled_range:
      curr_x = train_x[sample_row, :]
      curr_t = train_t[sample_row, :]
      w_prev = np.copy(w)

      w_star = gd_regularizer_error(w_prev, curr_x, curr_t, lambda_val, regularizer_func)
      w_change = eta * (t ** (-const.ALPHA)) * w_star
      w -= w_change
  return w


def build_lambdas():
  """
  Lambdas Builder

  Builds the values of lambda to test based off the widget slide.

  :return: Values of Lambda to test
  :rtype: List[float]
  """
  return [0] + [2 ** x for x in range(widgets.lambdas_range_slider.value[0],
                                      widgets.lambdas_range_slider.value[1] + 1)]


def _verify_eg_w_length(w):
  """
  EG Weights Verifier

  Debug only code.  Verifies that the weights of an EG weight
  vector are valid.

  :param w: Normalized EG weight vector
  :type w: np.ndarray
  """
  normalizer_err = abs(np.sum(w) - 1)
  assert normalizer_err < 10 ** -4
  # if normalizer_err > 10 ** -4:
  #     pass


def _build_random_results():
  """
  Debug Results Generator

  Builds debug results for testing while the program is being debugged.

  :return: Training, validation, and test errors in a list respectively.
  :rtype: List[np.ndarray]
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


def initialize_weights_gd(n):
  """
  GD Weight Vector Initializer

  Initializes all weights to zero in a Numpy matrix of the specified length.

  :param n: Number of dimensions in the weight vector
  :type n: int
  :return: Initial weight vector.
  :rtype: np.ndarray
  """
  return np.zeros([n, 1], dtype=np.float64)


def initialize_weights_eg(n):
  """
  EG Weight Vector Initializer

  Initializes weights to a random vector where the sum of the
  weights equals 1.

  :param n: Number of dimensions for the weight vector
  :type n: int

  :return: Normalized random array
  :rtype: np.array
  """
  rand_arr = np.random.rand(n, 1)
  sum_rand = np.sum(rand_arr)

  norm_rand = np.divide(rand_arr, sum_rand)
  _verify_eg_w_length(norm_rand)

  # Debug verification
  return norm_rand


def _extract_train_and_test_data(train_df, test_df, train_row_indexes=None, test_row_indexes=None,
                                 convert_to_plus_minus=False):
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
  :param convert_to_plus_minus: Creates a duplicsate copy of the minus data for use with EG+-
  :type convert_to_plus_minus: bool

  :return: Numpy matrices for trainX, trainT, testX, testT
  :rtype: List[np.ndarray]
  """
  train_x, train_t = _build_x_and_target(train_df, train_row_indexes,
                                         add_plus_minus=convert_to_plus_minus)

  test_x, test_t = _build_x_and_target(test_df, test_row_indexes,
                                       add_plus_minus=convert_to_plus_minus)

  return train_x, train_t, test_x, test_t


def _build_x_and_target(data_mat, row_indexes=None, add_plus_minus=False):
  """
  Pandas to NumPy Converter

  :param data_mat: Matrix of the data.  First column is the target data. The rest is feature data.
  :type data_mat: np.ndarray

  :param row_indexes: List of rows to selects.  Can be ignored if all the rows are desired.
  :type row_indexes: List[int]

  :return: Final learned weight vector
  :rtype: np.ndarray
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

  # Must be here as bias can be positive or negative
  if add_plus_minus:
    neg_values = np.multiply(-1, np.copy(x_values))
    x_values = np.hstack([x_values, neg_values])

  return x_values, target_values


def _build_dummy_dataset():
  """
  Dummy Dataset Builder

  Builds a fully linearly separable dataset for testing the learner.
  These datasets are then written to a file.
  """
  with open('AMtest.csv', 'r') as f:
    all_lines = [line.strip().split(",") for line in f.readlines()]

  a_strings = [line[1] for line in all_lines if line[1][0] == "a"][1:101]
  m_strings = [line[1] for line in all_lines if line[1][0] == "m"][1:101]

  file_names = ["coolest_guy_zayd_train.csv", "coolest_guy_zayd_test.csv"]
  for filename in file_names:
    with open(filename, "w") as f:
      for i in xrange(0, 1000):
        dict_strings = [a_strings, m_strings][np.random.randint(0, 2)]
        type_str = "ham" if dict_strings[1][0] == "a" else "spam"
        words = np.random.choice(dict_strings, size=10, replace=True)
        # noinspection PyTypeChecker
        f.write(",".join([type_str] + np.ndarray.tolist(words)) + "\n")


# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=50):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)

    Taken from: https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a

    Just prints a progress bar.  Used for debug purposes and not as a core feature
    or anything to do with the learning.
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    # noinspection PyByteLiteral,PyNonAsciiChar
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def build_results_filename():
  """
  Builds a filename based off the current widget settings,

  :return: Filename without path or extension
  :rtype: str
  """
  filename_str = "error_%s_%s_k=%d_alpha=%.1f_epochs=%d"
  filename = filename_str % (widgets.learning_alg_radio.value, widgets.regularizer_radio.value,
                             widgets.k_slider.value, widgets.learning_rate_slider.value,
                             widgets.epoch_slider.value)
  return filename.replace(" ", "_").replace("/", "").lower()


if __name__ == "__main__":
  widgets.learning_alg_radio.value = const.ALG_SGD
  widgets.epoch_slider.value = 10
  widgets.learning_rate_slider.value = 0.1
  widgets.error_type_radio.value = const.ERROR_ACCURACY
  widgets.k_slider.value = 5
  widgets.lambdas_range_slider.value = [-10, -8]
  widgets.learning_rate_slider.value = 0.2

  train_examples, test_examples = input_parser.parse()
  train_err_run, validation_err_run, test_err_run = run_hw03(train_examples, test_examples)
  import plotter
  plotter.display_plot = True
  plotter.create_plots(train_err_run, validation_err_run, test_err_run)

  import table_builder
  df = table_builder.create_table(train_err_run, validation_err_run, test_err_run)
  df.to_csv(build_results_filename() + ".csv", sep=",")
