import widgets
import random
import const
import input_parser


def run_hw03(train_data, test_data):
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
  lambdas = [1, 2, 3]  # TODO: Define how lambdas will be extracted

  # Build the indices for each of the folds.
  validation_sets = create_cross_validation_fold_sets(num_train, k)

  for lambda_val in lambdas:
    perform_cross_validation(train_data, validation_sets, test_data, learner_func, lambda_val)


def perform_cross_validation(train_data, validation_sets, test_data, learner_func, lambda_val):
  """

  :param train_data: Pandas DataFrame containing all TRAINING data with labels and features.
  :type train_data: pd.DataFrame

  :param validation_sets: Indices of the rows to be used for validation in each fold.
  :type validation_sets: List[List[int]]

  :param test_data: Pandas DataFrame containing all TESTING data with labels and features.
  :type test_data: pd.DataFrame

  :param learner_func:
  :type learner_func: callable

  :param lambda_val: Regularization value to be used
  :type lambda_val: float

  :return: Error mean and variable
  :rtype: List[float]
  """

  # Run cross validation
  k = len(validation_sets)
  training_sets = create_training_set_indices(validation_sets)

  for fold_cnt in xrange(0, k):
    print "Executing fold #" + str(fold_cnt)

    # Extract the training and test data
    train_slice = train_data.iloc[training_sets[fold_cnt]]
    valid_slice = train_data.iloc[validation_sets[fold_cnt]]

    learner_func(train_slice, valid_slice, lambda_val)

    print "Fold #" + str(fold_cnt) + " completed."

  # Train on the full training set then verify against the test data.
  learner_func(train_data, test_data, lambda_val)


def run_gradient_descent_learner(train_data, test_data, lambda_val):
  # TODO: Implement the gradient descent learner
  pass


def run_eg_learner(train_data, test_data, lambda_val):
  # TODO: Implement the EG learner
  pass


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


if __name__ == "__main__":
  train_examples, test_examples = input_parser.parse()
  run_hw03(train_examples, test_examples)
